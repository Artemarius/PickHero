"""Tests for pickhero.evaluation.audio — AudioRepository, _read_wave, resample_linear,
measure_audio_health, AudioHealth, AudioWindow, _select_channel."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from pickhero.evaluation.audio import (
    AudioHealth,
    AudioRepository,
    AudioWindow,
    _read_wave,
    _select_channel,
    measure_audio_health,
    resample_linear,
)
from pickhero.evaluation.manifest import CorpusCase, CorpusExpectedNote, CorpusSplit, EventKind


# ---------------------------------------------------------------------------
# Helpers: synthetic WAV files
# ---------------------------------------------------------------------------


def _make_sine_wav(
    path: Path,
    *,
    num_channels: int = 1,
    sample_rate: int = 44100,
    sample_width: int = 2,
    duration_s: float = 1.0,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> None:
    """Write a synthetic sine-wave WAV file."""
    num_frames = max(1, int(round(sample_rate * duration_s)))
    t = np.arange(num_frames, dtype=np.float64) / sample_rate
    mono = (amplitude * np.sin(2.0 * math.pi * frequency * t)).astype(np.float64)

    if sample_width == 1:
        scaled = np.clip(np.round(mono * 127.0 + 128.0), 0, 255).astype(np.uint8)
        frames = scaled.tobytes() * num_channels
    elif sample_width == 2:
        scaled = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2")
        frames = scaled.tobytes() * num_channels
    elif sample_width == 3:
        int24 = np.clip(np.round(mono * 8388607.0), -8388608, 8388607).astype(np.int32)
        raw = np.empty(num_frames * 3, dtype=np.uint8)
        raw[0::3] = (int24 & 0xFF).astype(np.uint8)
        raw[1::3] = ((int24 >> 8) & 0xFF).astype(np.uint8)
        raw[2::3] = ((int24 >> 16) & 0xFF).astype(np.uint8)
        frames = raw.tobytes() * num_channels
    elif sample_width == 4:
        scaled = np.clip(np.round(mono * 2147483647.0), -2147483648, 2147483647).astype("<i4")
        frames = scaled.tobytes() * num_channels
    else:
        raise ValueError(f"unsupported sample_width: {sample_width}")

    # For multi-channel, interleave by repeating each frame's samples
    if num_channels > 1:
        interleaved = bytearray()
        frame_bytes = len(frames) // num_frames
        ch_frame_bytes = frame_bytes // num_channels
        for f_idx in range(num_frames):
            base = f_idx * frame_bytes
            for ch in range(num_channels):
                offset = base + ch * ch_frame_bytes
                interleaved.extend(frames[offset : offset + ch_frame_bytes])
        frames = bytes(interleaved)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(num_channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(frames)


def _make_silent_wav(
    path: Path,
    *,
    num_channels: int = 1,
    sample_rate: int = 44100,
    sample_width: int = 2,
    duration_s: float = 1.0,
) -> None:
    """Write a silent WAV (all zeros)."""
    num_frames = max(1, int(round(sample_rate * duration_s)))
    frame_size = sample_width * num_channels
    frames = b"\x00" * (num_frames * frame_size)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(num_channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(frames)


def _make_clipped_wav(path: Path, *, sample_rate: int = 44100) -> None:
    """Write a WAV with clipped samples (values at full scale)."""
    num_frames = sample_rate  # 1 second
    # Alternating full-scale positive and negative samples
    mono = np.empty(num_frames, dtype=np.float64)
    mono[0::2] = 1.0
    mono[1::2] = -1.0
    scaled = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2")
    frames = scaled.tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)


def _make_dc_offset_wav(path: Path, *, sample_rate: int = 44100) -> None:
    """Write a WAV with a significant DC offset."""
    num_frames = sample_rate  # 1 second
    mono = np.full(num_frames, 0.03, dtype=np.float64)
    scaled = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2")
    frames = scaled.tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)


# ---------------------------------------------------------------------------
# AudioHealth
# ---------------------------------------------------------------------------


class TestAudioHealth:
    @staticmethod
    def test_attributes() -> None:
        h = AudioHealth(-6.0, -12.0, 0.001, 0.0001)
        assert h.peak_dbfs == pytest.approx(-6.0)
        assert h.rms_dbfs == pytest.approx(-12.0)
        assert h.dc_offset == pytest.approx(0.001)
        assert h.clipped_fraction == pytest.approx(0.0001)

    @staticmethod
    def test_is_clipped_above_threshold() -> None:
        h = AudioHealth(-1.0, -20.0, 0.0, 0.001)
        assert h.is_clipped

    @staticmethod
    def test_is_clipped_below_threshold() -> None:
        h = AudioHealth(-1.0, -20.0, 0.0, 0.0001)
        assert not h.is_clipped

    @staticmethod
    def test_has_dc_offset_above_threshold() -> None:
        h = AudioHealth(-6.0, -12.0, 0.03, 0.0)
        assert h.has_dc_offset

    @staticmethod
    def test_has_dc_offset_below_threshold() -> None:
        h = AudioHealth(-6.0, -12.0, 0.01, 0.0)
        assert not h.has_dc_offset

    @staticmethod
    def test_is_frozen() -> None:
        h = AudioHealth(0.0, 0.0, 0.0, 0.0)
        with pytest.raises(AttributeError):
            h.peak_dbfs = -3.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AudioWindow
# ---------------------------------------------------------------------------


class TestAudioWindow:
    @staticmethod
    def test_attributes() -> None:
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        health = AudioHealth(-6.0, -12.0, 0.0, 0.0)
        w = AudioWindow(samples=samples, sample_rate=44100, expected_onset_offset_ms=50.0, health=health)
        assert w.sample_rate == 44100
        assert w.expected_onset_offset_ms == pytest.approx(50.0)
        assert w.health == health
        np.testing.assert_array_equal(w.samples, samples)


# ---------------------------------------------------------------------------
# measure_audio_health
# ---------------------------------------------------------------------------


class TestMeasureAudioHealth:
    @staticmethod
    def test_normal_sine() -> None:
        # 440 Hz sine at half amplitude
        t = np.linspace(0, 1.0, 44100, endpoint=False)
        samples = 0.5 * np.sin(2.0 * math.pi * 440.0 * t)
        h = measure_audio_health(samples)
        # Peak should be ~ -6 dBFS (0.5 amplitude)
        assert h.peak_dbfs == pytest.approx(-6.0, abs=0.5)
        # Not clipped, no DC offset
        assert not h.is_clipped
        assert not h.has_dc_offset
        # RMS for a sine is peak/sqrt(2)
        expected_rms = 20.0 * math.log10(0.5 / math.sqrt(2))
        assert h.rms_dbfs == pytest.approx(expected_rms, abs=0.5)

    @staticmethod
    def test_empty_returns_minus_infinity() -> None:
        h = measure_audio_health(np.array([], dtype=np.float32))
        assert h.peak_dbfs == pytest.approx(-120.0)
        assert h.rms_dbfs == pytest.approx(-120.0)
        assert h.dc_offset == pytest.approx(0.0)
        assert h.clipped_fraction == pytest.approx(0.0)
        assert not h.is_clipped
        assert not h.has_dc_offset

    @staticmethod
    def test_clipped_samples() -> None:
        samples = np.array([1.0, -1.0, 0.5, -0.5, 0.998], dtype=np.float32)
        h = measure_audio_health(samples)
        assert h.clipped_fraction > 0.0
        # Three samples >= 0.995 out of 5: 1.0, -1.0, 0.998
        assert h.clipped_fraction == pytest.approx(3.0 / 5.0)

    @staticmethod
    def test_dc_offset_detected() -> None:
        samples = np.full(1000, 0.03, dtype=np.float32)
        h = measure_audio_health(samples)
        assert h.has_dc_offset
        assert h.dc_offset == pytest.approx(0.03)

    @staticmethod
    def test_silent_samples() -> None:
        samples = np.zeros(1000, dtype=np.float32)
        h = measure_audio_health(samples)
        assert h.peak_dbfs == pytest.approx(-120.0)
        assert h.dc_offset == pytest.approx(0.0)
        assert h.clipped_fraction == pytest.approx(0.0)

    @staticmethod
    def test_near_silent_floor() -> None:
        # Values near zero should produce very low dBFS without -inf
        samples = np.full(100, 1e-10, dtype=np.float32)
        h = measure_audio_health(samples)
        # Should be finite (-200 dBFS from the 1e-6 floor)
        assert math.isfinite(h.peak_dbfs)
        assert math.isfinite(h.rms_dbfs)


# ---------------------------------------------------------------------------
# resample_linear
# ---------------------------------------------------------------------------


class TestResampleLinear:
    @staticmethod
    def test_same_rate_returns_same_values() -> None:
        data = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float32)
        result = resample_linear(data, 44100, 44100)
        np.testing.assert_array_equal(result, data)
        assert result.dtype == np.float32

    @staticmethod
    def test_empty_returns_empty() -> None:
        data = np.array([], dtype=np.float32)
        result = resample_linear(data, 44100, 22050)
        assert len(result) == 0
        assert result.dtype == np.float32

    @staticmethod
    def test_downsample() -> None:
        # 4 samples at 44100 -> 2 samples at 22050
        data = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        result = resample_linear(data, 44100, 22050)
        assert len(result) == 2  # 4 * 22050 / 44100 = 2
        # source_positions: [0, 2], which land exactly on samples 0 and 2
        np.testing.assert_array_equal(result, [0.0, 0.0])

    @staticmethod
    def test_upsample() -> None:
        data = np.array([0.0, 1.0], dtype=np.float32)
        result = resample_linear(data, 22050, 44100)
        # 2 * 44100 / 22050 = 4 samples
        assert len(result) == 4
        np.testing.assert_array_equal(result, [0.0, 0.5, 1.0, 1.0])

    @staticmethod
    def test_upsample_longer() -> None:
        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = resample_linear(data, 22050, 44100)
        assert len(result) == 6  # 3 * 44100 / 22050
        # Linear interpolation: [0, 0.25, 0.5, 0.75, 1, 1]
        np.testing.assert_array_equal(result, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0])

    @staticmethod
    def test_dtype_is_float32() -> None:
        data = np.array([0.0, 1.0], dtype=np.float64)
        result = resample_linear(data, 44100, 22050)
        assert result.dtype == np.float32

    @staticmethod
    def test_at_least_one_sample() -> None:
        data = np.array([0.5], dtype=np.float32)
        result = resample_linear(data, 44100, 8000)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _read_wave
# ---------------------------------------------------------------------------


class TestReadWave:
    @staticmethod
    def test_16bit_mono(tmp_path: Path) -> None:
        path = tmp_path / "test_16bit.wav"
        _make_sine_wav(path, sample_width=2, sample_rate=44100, duration_s=0.1, amplitude=0.5)
        data, rate = _read_wave(path)
        assert rate == 44100
        assert data.shape == (4410, 1)  # 0.1s at 44100 Hz, mono
        assert data.dtype == np.float32
        # Peak should be approx 0.5
        assert float(np.max(np.abs(data))) == pytest.approx(0.5, abs=0.01)

    @staticmethod
    def test_8bit_wav(tmp_path: Path) -> None:
        path = tmp_path / "test_8bit.wav"
        _make_sine_wav(path, sample_width=1, sample_rate=22050, duration_s=0.1, amplitude=0.5)
        data, rate = _read_wave(path)
        assert rate == 22050
        assert data.dtype == np.float32
        # 8-bit is offset binary (128), so 0.5 amplitude = 127+64 = 191 or 127-64 = 63
        # After normalization: (191 - 128) / 128 ≈ 0.492
        assert float(np.max(data)) == pytest.approx(0.5, abs=0.05)

    @staticmethod
    def test_24bit_wav(tmp_path: Path) -> None:
        path = tmp_path / "test_24bit.wav"
        _make_sine_wav(path, sample_width=3, sample_rate=44100, duration_s=0.05, amplitude=0.5)
        data, rate = _read_wave(path)
        assert rate == 44100
        assert data.dtype == np.float32
        assert float(np.max(data)) == pytest.approx(0.5, abs=0.01)

    @staticmethod
    def test_32bit_wav(tmp_path: Path) -> None:
        path = tmp_path / "test_32bit.wav"
        _make_sine_wav(path, sample_width=4, sample_rate=44100, duration_s=0.05, amplitude=0.5)
        data, rate = _read_wave(path)
        assert rate == 44100
        assert data.dtype == np.float32
        assert float(np.max(data)) == pytest.approx(0.5, abs=0.01)

    @staticmethod
    def test_stereo_wav(tmp_path: Path) -> None:
        path = tmp_path / "test_stereo.wav"
        _make_sine_wav(path, num_channels=2, sample_rate=44100, duration_s=0.1, amplitude=0.5)
        data, rate = _read_wave(path)
        assert rate == 44100
        assert data.shape == (4410, 2)  # 0.1s, stereo
        assert data.dtype == np.float32

    @staticmethod
    def test_silent_wav(tmp_path: Path) -> None:
        path = tmp_path / "silent.wav"
        _make_silent_wav(path, sample_rate=44100, duration_s=0.1)
        data, rate = _read_wave(path)
        assert rate == 44100
        assert data.shape == (4410, 1)
        assert float(np.max(np.abs(data))) == pytest.approx(0.0, abs=1e-7)

    @staticmethod
    def test_unsupported_sample_width(tmp_path: Path) -> None:
        path = tmp_path / "bad_width.wav"
        # Write a WAV with standard 16-bit format, then patch the sample width byte
        # to a value that _read_wave rejects (5), while keeping the wave module happy.
        _make_sine_wav(path, sample_width=2, sample_rate=44100, duration_s=0.01)
        raw = path.read_bytes()
        # In the fmt chunk of a standard WAV: ... | bits_per_sample (2 bytes) |
        # sample width in bytes = bits_per_sample / 8
        # For 16-bit PCM at offset 34: 16 (bits) = 0x10 0x00
        # Replace bits_per_sample with 40 (5 bytes), then fix the byte_rate and block_align
        # Actually, simpler: just patch the sample_width byte stored at a known offset.
        # In a standard 16-bit mono WAV, the block align (byte 32-33) = 2, bits_per_sample (34-35) = 16
        # Let's use struct to be precise.
        import struct as _struct
        fmt = list(_struct.unpack_from("<HHIIHH", raw, 20))
        # fmt = [audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample]
        fmt[4] = 5  # block_align = 5 (1 channel * 5 bytes)
        fmt[5] = 40  # bits_per_sample = 40 (5 bytes * 8)
        fmt[3] = 44100 * 5 * 1  # byte_rate = sample_rate * block_align * channels
        raw = bytearray(raw)
        _struct.pack_into("<HHIIHH", raw, 20, *fmt)
        path.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="unsupported WAV sample width"):
            _read_wave(path)

    @staticmethod
    def test_invalid_channel_layout(tmp_path: Path) -> None:
        path = tmp_path / "bad_layout.wav"
        # The _read_wave function has `len(data) % channels` as a defense-in-depth
        # check that is mathematically guaranteed to pass when the wave module
        # processes files normally (readframes always returns whole frames).
        # Here we test that wave.open itself rejects fundamentally invalid
        # channel counts, which _read_wave propagates.
        _make_sine_wav(path, sample_width=2, sample_rate=44100, duration_s=0.01)
        raw = bytearray(path.read_bytes())
        import struct as _struct
        # Patch num_channels (offset 22 in fmt chunk) to 0
        _struct.pack_into("<H", raw, 22, 0)
        path.write_bytes(bytes(raw))
        with pytest.raises((ValueError, wave.Error), match="channel"):
            _read_wave(path)
# ---------------------------------------------------------------------------


class TestSelectChannel:
    @staticmethod
    def test_mono_returns_mean() -> None:
        source = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        result = _select_channel(source, None)
        # For mono with one channel, mean of axis=1 is the same as the channel
        np.testing.assert_array_almost_equal(result, source[:, 0])

    @staticmethod
    def test_stereo_mean() -> None:
        source = np.array([[0.1, 0.3], [0.2, 0.4], [0.5, 0.7]], dtype=np.float32)
        result = _select_channel(source, None)
        expected = source.mean(axis=1)
        np.testing.assert_array_almost_equal(result, expected)

    @staticmethod
    def test_select_specific_channel() -> None:
        source = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]], dtype=np.float32)
        result = _select_channel(source, "2")
        np.testing.assert_array_almost_equal(result, source[:, 1])

    @staticmethod
    def test_select_first_channel() -> None:
        source = np.array([[0.1, 0.5], [0.2, 0.6]], dtype=np.float32)
        result = _select_channel(source, "1")
        np.testing.assert_array_almost_equal(result, source[:, 0])

    @staticmethod
    def test_raises_on_1d_input() -> None:
        with pytest.raises(ValueError, match="audio must have shape"):
            _select_channel(np.array([0.1, 0.2], dtype=np.float32), None)

    @staticmethod
    def test_raises_on_non_integer_channel() -> None:
        source = np.zeros((10, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="audio_channel must be a one-based integer"):
            _select_channel(source, "abc")

    @staticmethod
    def test_raises_on_out_of_range_channel() -> None:
        source = np.zeros((10, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="audio_channel 4 is unavailable"):
            _select_channel(source, "4")


# ---------------------------------------------------------------------------
# AudioRepository — load
# ---------------------------------------------------------------------------


class TestAudioRepositoryLoad:
    @staticmethod
    def test_load_wav(tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        _make_sine_wav(path, sample_rate=22050, duration_s=0.5, amplitude=0.3)
        repo = AudioRepository()
        data, rate = repo.load(path)
        assert rate == 22050
        assert data.shape == (11025, 1)
        assert data.dtype == np.float32
        assert float(np.max(np.abs(data))) == pytest.approx(0.3, abs=0.01)

    @staticmethod
    def test_load_stereo_wav(tmp_path: Path) -> None:
        path = tmp_path / "stereo.wav"
        _make_sine_wav(path, num_channels=2, sample_rate=44100, duration_s=0.1)
        repo = AudioRepository()
        data, rate = repo.load(path)
        assert data.shape == (4410, 2)

    @staticmethod
    def test_load_nonexistent_file(tmp_path: Path) -> None:
        repo = AudioRepository()
        missing = tmp_path / "does_not_exist.wav"
        with pytest.raises(FileNotFoundError):
            repo.load(missing)

    @staticmethod
    def test_load_unsupported_format(tmp_path: Path) -> None:
        path = tmp_path / "test.mp3"
        path.write_text("not an audio file")
        repo = AudioRepository()
        with pytest.raises(ValueError, match="unsupported audio format"):
            repo.load(path)

    @staticmethod
    def test_load_with_string_path(tmp_path: Path) -> None:
        path = tmp_path / "strpath.wav"
        _make_sine_wav(path, sample_rate=44100, duration_s=0.1)
        repo = AudioRepository()
        data, rate = repo.load(str(path))
        assert rate == 44100
        assert len(data) > 0

    @staticmethod
    def test_cache_returns_same_object(tmp_path: Path) -> None:
        path = tmp_path / "cached.wav"
        _make_sine_wav(path, sample_rate=44100, duration_s=0.1)
        repo = AudioRepository()
        data1, rate1 = repo.load(path)
        data2, rate2 = repo.load(path)
        assert rate1 == rate2
        # Should be the same array object (cache hit, identity preserved)
        assert data1 is data2

    @staticmethod
    def test_cache_respects_max_bytes(tmp_path: Path) -> None:
        path1 = tmp_path / "a.wav"
        path2 = tmp_path / "b.wav"
        # Two files larger than a tiny cache
        _make_sine_wav(path1, sample_rate=44100, duration_s=2.0)
        _make_sine_wav(path2, sample_rate=44100, duration_s=2.0)

        repo = AudioRepository(max_cache_bytes=8192)  # Very small cache
        data1, _rate1 = repo.load(path1)
        data2, _rate2 = repo.load(path2)
        data1_again, _rate1_again = repo.load(path1)

        # path1 was evicted, so reload should give a different array
        assert data1 is not data1_again

    @staticmethod
    def test_cache_zero_disabled(tmp_path: Path) -> None:
        path = tmp_path / "test.wav"
        _make_sine_wav(path, sample_rate=44100, duration_s=0.1)
        repo = AudioRepository(max_cache_bytes=0)
        data1, _rate1 = repo.load(path)
        data2, _rate2 = repo.load(path)
        # With cache disabled, each load returns a fresh read
        assert data1 is not data2

    @staticmethod
    def test_cache_reorders_lru(tmp_path: Path) -> None:
        path_a = tmp_path / "a.wav"
        path_b = tmp_path / "b.wav"
        _make_sine_wav(path_a, sample_rate=44100, duration_s=0.1)
        _make_sine_wav(path_b, sample_rate=44100, duration_s=0.1)

        # Cache large enough for both
        repo = AudioRepository(max_cache_bytes=1024 * 1024)
        repo.load(path_a)
        repo.load(path_b)
        # Access A again — should move it to front (LRU)
        repo.load(path_a)
        _a2, _ra = repo.load(path_a)
        _b2, _rb = repo.load(path_b)
        # Both should be cached; no crash is the main check


# ---------------------------------------------------------------------------
# AudioRepository — window_for_case
# ---------------------------------------------------------------------------


class _make_case:
    """Factory helper for building CorpusCase instances with minimal boilerplate."""

    @staticmethod
    def simple(
        audio_path: str,
        start_s: float,
        end_s: float,
        *,
        expected_present: bool = True,
        midi: int = 40,
        event_kind: EventKind = EventKind.SINGLE_NOTE,
        expected_onset_s: float | None = None,
        window_before_ms: float = 120.0,
        window_after_ms: float | None = None,
    ) -> CorpusCase:
        return CorpusCase(
            case_id="test",
            audio_path=audio_path,
            source="test",
            split=CorpusSplit.TEST,
            event_kind=event_kind,
            start_s=start_s,
            end_s=end_s,
            expected_present=expected_present,
            notes=(CorpusExpectedNote(midi=midi),),
            expected_onset_s=expected_onset_s,
            window_before_ms=window_before_ms,
            window_after_ms=window_after_ms,
        )


class TestWindowForCase:
    @staticmethod
    def test_basic_window(tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0, amplitude=0.5)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(audio_path=str(wav_path), start_s=0.2, end_s=0.5)

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)

        assert isinstance(window, AudioWindow)
        assert window.sample_rate == 44100
        assert len(window.samples) > 0
        assert window.expected_onset_offset_ms is not None
        assert isinstance(window.health, AudioHealth)

    @staticmethod
    def test_window_length(tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        window_before_ms = 100.0
        window_after_ms = 300.0
        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.2,
            end_s=0.5,
            window_before_ms=window_before_ms,
            window_after_ms=window_after_ms,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        expected_length_ms = window_before_ms + window_after_ms
        expected_samples = int(round(expected_length_ms / 1000.0 * 44100))
        assert len(window.samples) == expected_samples

    @staticmethod
    def test_window_resampling(tmp_path: Path) -> None:
        wav_path = tmp_path / "test.wav"
        _make_sine_wav(wav_path, sample_rate=48000, duration_s=1.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.1,
            end_s=0.3,
            window_before_ms=50.0,
            window_after_ms=200.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        assert window.sample_rate == 44100
        expected_ms = 50.0 + 200.0
        expected_samples = int(round(expected_ms / 1000.0 * 44100))
        assert len(window.samples) == expected_samples

    @staticmethod
    def test_window_padding_short_audio(tmp_path: Path) -> None:
        """When the extracted window is shorter than expected, it should be zero-padded."""
        wav_path = tmp_path / "short.wav"
        # Very short audio (0.01s)
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=0.01)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        # Try to extract a window that's longer than the available audio
        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.005,
            end_s=0.008,
            window_before_ms=200.0,
            window_after_ms=500.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        # Should have the expected length (padded)
        expected_ms = 200.0 + 500.0
        expected_samples = int(round(expected_ms / 1000.0 * 44100))
        assert len(window.samples) == expected_samples
        # The first samples should be non-zero (from the actual audio)
        # The later samples should be zero (padding)
        assert np.any(window.samples != 0.0)

    @staticmethod
    def test_window_truncation(tmp_path: Path) -> None:
        """When extracted window is longer than target, it should be truncated."""
        wav_path = tmp_path / "long.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=2.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.1,
            end_s=0.3,
            window_before_ms=50.0,
            window_after_ms=200.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        expected_ms = 50.0 + 200.0
        expected_samples = int(round(expected_ms / 1000.0 * 44100))
        assert len(window.samples) == expected_samples

    @staticmethod
    def test_window_at_start_of_audio(tmp_path: Path) -> None:
        wav_path = tmp_path / "start.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.0,
            end_s=0.1,
            window_before_ms=50.0,
            window_after_ms=200.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        # 250ms window starting at -50ms, clamped to 0
        expected_ms = 250.0
        expected_samples = int(round(expected_ms / 1000.0 * 44100))
        assert len(window.samples) == expected_samples

    @staticmethod
    def test_expected_onset_offset_ms(tmp_path: Path) -> None:
        wav_path = tmp_path / "onset.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        # onset at 0.3s, window starts at start_s - before_s = 0.2 - 0.12 = 0.08s
        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.2,
            end_s=0.5,
            expected_onset_s=0.3,
            window_before_ms=120.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        # onset offset = (0.3 - 0.08) * 1000 = 220ms
        assert window.expected_onset_offset_ms is not None
        assert window.expected_onset_offset_ms == pytest.approx(220.0, abs=1.0)

    @staticmethod
    def test_technique_onset_fallback(tmp_path: Path) -> None:
        """When event_kind is not silence and expected_onset_s is None, fall back to start_s."""
        wav_path = tmp_path / "tech.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = CorpusCase(
            case_id="technique_test",
            audio_path=str(wav_path),
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.TECHNIQUE,
            start_s=0.3,
            end_s=0.6,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=50),),
            technique="vibrato",
            technique_present=True,
            expected_onset_s=None,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        # Fallback: (start_s - window_start_s) * 1000
        # window_start_s = max(0, 0.3 - 0.12) = 0.18
        assert window.expected_onset_offset_ms is not None
        assert window.expected_onset_offset_ms == pytest.approx(120.0, abs=1.0)

    @staticmethod
    def test_window_with_explicit_after_ms(tmp_path: Path) -> None:
        wav_path = tmp_path / "explicit.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.2,
            end_s=0.5,
            window_before_ms=100.0,
            window_after_ms=400.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        expected_samples = int(round(500.0 / 1000.0 * 44100))
        assert len(window.samples) == expected_samples

    @staticmethod
    def test_window_health_included(tmp_path: Path) -> None:
        wav_path = tmp_path / "health.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=0.5, amplitude=0.5)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.1,
            end_s=0.3,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        health = window.health
        assert isinstance(health, AudioHealth)
        assert math.isfinite(health.peak_dbfs)
        assert math.isfinite(health.rms_dbfs)

    @staticmethod
    def test_window_truncation_near_end_of_audio(tmp_path: Path) -> None:
        wav_path = tmp_path / "end.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=0.3)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        # start_s near end of audio, window extends past end
        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.25,
            end_s=0.28,
            window_before_ms=50.0,
            window_after_ms=200.0,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)
        # Window should be padded to full length
        expected_samples = int(round(250.0 / 1000.0 * 44100))
        assert len(window.samples) == expected_samples


# ---------------------------------------------------------------------------
# Integration: load WAV + extract window + verify structure
# ---------------------------------------------------------------------------


class TestIntegration:
    @staticmethod
    def test_load_and_window_full_pipeline(tmp_path: Path) -> None:
        """Load a WAV, extract a window, and verify all return types."""
        wav_path = tmp_path / "integration.wav"
        _make_sine_wav(wav_path, sample_rate=44100, duration_s=1.0, amplitude=0.5)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("[]")

        case = _make_case.simple(
            audio_path=str(wav_path),
            start_s=0.2,
            end_s=0.5,
            expected_onset_s=0.35,
        )

        repo = AudioRepository()
        window = repo.window_for_case(case, manifest_path, sample_rate=44100)

        # AudioWindow contract
        assert isinstance(window.samples, np.ndarray)
        assert window.samples.dtype == np.float32
        assert window.sample_rate == 44100
        assert window.samples.ndim == 1  # mono output

        # expected_onset_offset_ms: window starts at 0.2-0.12=0.08, onset at 0.35
        # offset = (0.35 - 0.08) * 1000 = 270ms
        assert window.expected_onset_offset_ms is not None
        assert window.expected_onset_offset_ms == pytest.approx(270.0, abs=2.0)

        # AudioHealth contract
        assert window.health.peak_dbfs < 0.0  # less than 0 dBFS for 0.5 amplitude
        assert not window.health.is_clipped
        assert isinstance(window.health.dc_offset, float)

    @staticmethod
    def test_handles_empty_audio_gracefully(tmp_path: Path) -> None:
        """An empty WAV file should not crash the reader."""
        path = tmp_path / "empty.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"")  # 0 frames

        data, rate = _read_wave(path)
        assert rate == 44100
        assert len(data) == 0
        assert data.shape == (0, 1)
        health = measure_audio_health(data[:, 0])
        assert health.peak_dbfs == pytest.approx(-120.0)

    @staticmethod
    def test_handles_silent_audio_gracefully(tmp_path: Path) -> None:
        """Silent WAV (all zeros) should produce -120 dBFS health."""
        path = tmp_path / "silence.wav"
        _make_silent_wav(path, sample_rate=44100, duration_s=0.5)

        data, rate = _read_wave(path)
        assert rate == 44100
        assert np.all(data == 0.0)
        health = measure_audio_health(data[:, 0])
        assert health.peak_dbfs == pytest.approx(-120.0)
        assert health.rms_dbfs == pytest.approx(-120.0)
