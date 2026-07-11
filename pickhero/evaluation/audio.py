"""Deterministic audio loading and event window extraction for evaluation."""

from __future__ import annotations

import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pickhero.evaluation.manifest import CorpusCase


@dataclass(frozen=True)
class AudioHealth:
    peak_dbfs: float
    rms_dbfs: float
    dc_offset: float
    clipped_fraction: float

    @property
    def is_clipped(self) -> bool:
        return self.clipped_fraction >= 0.0005

    @property
    def has_dc_offset(self) -> bool:
        return abs(self.dc_offset) >= 0.02


@dataclass(frozen=True)
class AudioWindow:
    samples: np.ndarray
    sample_rate: int
    expected_onset_offset_ms: float | None
    health: AudioHealth


class AudioRepository:
    """Bounded audio cache with correct source-rate and channel slicing."""

    def __init__(self, max_cache_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_cache_bytes = max(0, int(max_cache_bytes))
        self._cache: OrderedDict[Path, tuple[np.ndarray, int]] = OrderedDict()
        self._cache_bytes = 0

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        """Return float32 audio shaped ``(frames, channels)``."""
        resolved = Path(path).expanduser().resolve()
        cached = self._cache.pop(resolved, None)
        if cached is not None:
            self._cache[resolved] = cached
            return cached
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        suffix = resolved.suffix.lower()
        if suffix == ".wav":
            result = _read_wave(resolved)
        elif suffix in {".flac", ".ogg", ".aiff", ".aif"}:
            result = _read_soundfile(resolved)
        else:
            raise ValueError(f"unsupported audio format: {resolved.suffix}")
        self._remember(resolved, result)
        return result

    def _remember(self, path: Path, value: tuple[np.ndarray, int]) -> None:
        samples, _sample_rate = value
        size = int(samples.nbytes)
        if self.max_cache_bytes <= 0 or size > self.max_cache_bytes:
            return
        while self._cache and self._cache_bytes + size > self.max_cache_bytes:
            _old_path, (old_samples, _old_rate) = self._cache.popitem(last=False)
            self._cache_bytes -= int(old_samples.nbytes)
        self._cache[path] = value
        self._cache_bytes += size

    def window_for_case(
        self,
        case: CorpusCase,
        manifest_path: str | Path,
        sample_rate: int,
    ) -> AudioWindow:
        path = case.resolve_audio_path(Path(manifest_path))
        source, source_rate = self.load(path)
        mono = _select_channel(source, case.metadata.get("audio_channel"))

        before_s = max(0.0, case.window_before_ms / 1000.0)
        duration_s = case.end_s - case.start_s
        if case.window_after_ms is None:
            # Techniques and sustains need the authored duration; short attacks
            # still get enough post-roll for stable spectral evidence.
            after_s = max(0.32, duration_s + 0.12)
        else:
            after_s = max(0.0, case.window_after_ms / 1000.0)

        window_start_s = max(0.0, case.start_s - before_s)
        window_end_s = min(len(mono) / source_rate, case.start_s + after_s)
        start_sample = max(0, int(round(window_start_s * source_rate)))
        end_sample = min(len(mono), int(round(window_end_s * source_rate)))
        segment = mono[start_sample:end_sample]
        if source_rate != sample_rate:
            segment = resample_linear(segment, source_rate, sample_rate)

        target_length = max(1, int(round((before_s + after_s) * sample_rate)))
        if len(segment) < target_length:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[: len(segment)] = segment
            segment = padded
        elif len(segment) > target_length:
            segment = segment[:target_length]

        expected_onset_offset_ms: float | None = None
        if case.expected_onset_s is not None:
            expected_onset_offset_ms = (
                case.expected_onset_s - window_start_s
            ) * 1000.0
        elif case.event_kind.value != "silence":
            expected_onset_offset_ms = (case.start_s - window_start_s) * 1000.0

        return AudioWindow(
            samples=segment.astype(np.float32, copy=False),
            sample_rate=sample_rate,
            expected_onset_offset_ms=expected_onset_offset_ms,
            health=measure_audio_health(segment),
        )


def _select_channel(source: np.ndarray, channel_value: str | None) -> np.ndarray:
    if source.ndim != 2 or source.shape[1] < 1:
        raise ValueError(f"audio must have shape (frames, channels), got {source.shape}")
    if channel_value:
        try:
            channel_index = int(channel_value) - 1
        except ValueError as exc:
            raise ValueError(f"audio_channel must be a one-based integer, got {channel_value!r}") from exc
        if channel_index < 0 or channel_index >= source.shape[1]:
            raise ValueError(
                f"audio_channel {channel_index + 1} is unavailable; file has {source.shape[1]} channel(s)"
            )
        return source[:, channel_index]
    return source.mean(axis=1)


def _read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frames = stream.readframes(stream.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        values = np.where(values >= 2**23, values - 2**24, values)
        data = values.astype(np.float32) / float(2**23)
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / float(2**31)
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels < 1 or len(data) % channels:
        raise ValueError(f"invalid WAV channel layout in {path}")
    return data.reshape(-1, channels).astype(np.float32, copy=False), sample_rate


def _read_soundfile(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            f"reading {path.suffix} requires the optional 'evaluation' dependencies"
        ) from exc
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return data.astype(np.float32, copy=False), int(sample_rate)


def resample_linear(data: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
    if old_rate == new_rate or len(data) == 0:
        return data.astype(np.float32, copy=False)
    new_length = max(1, int(round(len(data) * new_rate / old_rate)))
    source_positions = np.arange(new_length, dtype=np.float64) * old_rate / new_rate
    left = np.floor(source_positions).astype(np.intp)
    right = np.minimum(left + 1, len(data) - 1)
    fraction = (source_positions - left).astype(np.float32)
    return (data[left] * (1.0 - fraction) + data[right] * fraction).astype(np.float32)


def measure_audio_health(samples: np.ndarray) -> AudioHealth:
    if len(samples) == 0:
        return AudioHealth(-120.0, -120.0, 0.0, 0.0)
    absolute = np.abs(samples)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return AudioHealth(
        peak_dbfs=20.0 * np.log10(max(peak, 1e-6)),
        rms_dbfs=20.0 * np.log10(max(rms, 1e-6)),
        dc_offset=float(np.mean(samples, dtype=np.float64)),
        clipped_fraction=float(np.mean(absolute >= 0.995)),
    )
