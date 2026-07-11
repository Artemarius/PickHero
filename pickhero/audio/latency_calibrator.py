"""Output-to-input round-trip latency measurement.

Supports two measurement modes:

1. **Electrical loopback** (default, when a loopback cable is detected):
   Plays a short click through the default output and records it on the
   selected input channel.  A cross-correlation peak >= 0.8 qualifies as
   electrical loopback — the cable connects output directly to input so
   minimal acoustic energy is lost.  Confidence threshold for acceptance:
   0.8.

2. **Acoustic** (fallback):
   Same measurement but assuming the sound travels through the air.
   Confidence threshold for acceptance: 0.6.

If no output device is available the returned result is always
``method="no_output"`` with ``accepted=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _device_kind(device: dict | object) -> str:
    """Return a short kind string for an input device dict."""
    if isinstance(device, dict):
        try:
            return str(device.get("hostApi", "?"))
        except Exception:
            return "?"
    return "?"


@dataclass
class LatencyResult:
    delay_ms: float
    confidence: float
    accepted: bool
    method: str

    def to_dict(self) -> dict:
        return {
            "delay_ms": float(self.delay_ms),
            "confidence": float(self.confidence),
            "accepted": bool(self.accepted),
            "method": str(self.method),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_click(
    sample_rate: int,
    click_duration: float = 0.05,
    record_duration: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a raised-cosine click and a full-length output buffer.

    Returns ``(click, output)`` where *output* is zero-filled silence long
    enough for *record_duration* with the click placed at the very start.
    Both arrays are ``float32``.
    """
    click_samples = int(sample_rate * click_duration)
    record_samples = int(sample_rate * record_duration)
    click = np.hanning(click_samples).astype(np.float32)
    output = np.zeros(record_samples, dtype=np.float32)
    output[:click_samples] = click
    return click, output


def _resolve_input_device(
    input_device: int | str | None,
) -> int:
    """Resolve an opaque input device identifier to a PortAudio device index.

    Falls back to ``sounddevice.default.device[0]`` on failure.
    """
    import sounddevice as sd

    try:
        if input_device is None:
            return int(sd.default.device[0])
        if isinstance(input_device, int):
            sd.query_devices(input_device)
            return input_device
        # String match against device name.
        devices = sd.query_devices()
        matches = [
            i
            for i, d in enumerate(devices)
            if input_device in (d.get("name") or "")
        ]
        return int(matches[0]) if matches else int(sd.default.device[0])
    except Exception:
        return int(sd.default.device[0])


def _resolve_output_device() -> int | None:
    """Return the default output device index, or *None* if unavailable."""
    import sounddevice as sd

    try:
        idx = sd.default.device[1]
        if idx is None:
            return None
        sd.query_devices(idx)
        return int(idx)
    except Exception:
        return None


def _playrec_and_correlate(
    output: np.ndarray,
    click: np.ndarray,
    sample_rate: int,
    input_idx: int,
    output_idx: int,
    input_channel: int,
) -> tuple[float, float]:
    """Full-duplex play/record and cross-correlate.

    Returns ``(delay_ms, confidence)`` where *confidence* is the normalised
    cross-correlation peak (0.0 – 1.0).  On any error (playrec failure, short
    recording, …) returns ``(0.0, 0.0)`` — the caller decides acceptance.
    """
    import sounddevice as sd

    sample_rate = int(sample_rate)

    # Determine channel count for the input device.
    try:
        dev_info = sd.query_devices(input_idx)
        max_input_channels = int(dev_info.get("max_input_channels", 1))
    except Exception:
        max_input_channels = 1

    if max_input_channels <= 0:
        return 0.0, 0.0

    ch = max(0, min(int(input_channel), max_input_channels - 1))

    try:
        recorded = sd.playrec(
            output,
            samplerate=sample_rate,
            channels=max_input_channels,
            device=(input_idx, output_idx),
            dtype=np.float32,
        )
        sd.wait()
    except Exception:
        return 0.0, 0.0

    if recorded.ndim == 1:
        input_signal = recorded
    else:
        input_signal = recorded[:, ch]

    if len(input_signal) < len(click):
        return 0.0, 0.0

    # Normalised cross-correlation.
    input_norm = input_signal - np.mean(input_signal)
    click_norm = click - np.mean(click)
    denom = np.sqrt(np.sum(input_norm**2) * np.sum(click_norm**2)) + 1e-15
    correlation = np.correlate(input_norm, click_norm, mode="full") / denom
    peak_idx = int(np.argmax(correlation))
    peak_value = float(correlation[peak_idx])

    delay_samples = peak_idx - (len(click_norm) - 1)
    delay_ms = max(0.0, float(delay_samples) * 1000.0 / sample_rate)

    return delay_ms, max(0.0, min(1.0, peak_value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_loopback_connection(
    sample_rate: int,
    input_device: int | str | None,
    input_channel: int,
) -> bool:
    """Check whether an electrical loopback cable is connected.

    Plays a short test click through the default output and records it.
    If the normalised cross-correlation peak is >= 0.8 the signal is assumed
    to have travelled through a direct cable (electrical loopback) rather
    than through the air.  Returns ``True`` when loopback is detected.

    When no output device is available this function returns ``False``
    without raising.
    """
    sample_rate = int(sample_rate)
    input_channel = int(input_channel)

    input_idx = _resolve_input_device(input_device)
    output_idx = _resolve_output_device()
    if output_idx is None:
        return False

    click, output = _build_click(sample_rate)
    delay_ms, confidence = _playrec_and_correlate(
        output, click, sample_rate, input_idx, output_idx, input_channel,
    )
    # confidence >= 0.8 is the electrical-loopback threshold.
    return confidence >= 0.8 and delay_ms < 0.6 * 1000.0


def measure_roundtrip_latency(
    sample_rate: int,
    input_device: int | str | None,
    input_channel: int,
) -> LatencyResult:
    """Measure output-to-input latency in milliseconds.

    Plays a short click through the default output and records the input.
    Returns a ``LatencyResult`` with delay, confidence, and acceptance status.

    Strategy
    --------
    1. If no output device is available → ``method="no_output"``, not accepted.
    2. Try **electrical loopback** measurement (confidence threshold 0.8).
       If the correlation peak >= 0.8 the measurement is accepted as
       ``method="electrical_loopback"``.
    3. Fall back to **acoustic** measurement (confidence threshold 0.6).
       If the correlation peak >= 0.6 the measurement is accepted as
       ``method="acoustic"``.
    4. If neither threshold is met the result is not accepted (method
       reflects the last attempted mode, ``"acoustic"``).
    """
    import sounddevice as sd  # noqa: F401 — ensure available

    sample_rate = int(sample_rate)
    input_channel = int(input_channel)

    # Resolve input / output devices.
    input_idx = _resolve_input_device(input_device)
    output_idx = _resolve_output_device()

    if output_idx is None:
        return LatencyResult(
            delay_ms=0.0,
            confidence=0.0,
            accepted=False,
            method="no_output",
        )

    click, output = _build_click(sample_rate)
    record_duration_ms = len(output) * 1000.0 / sample_rate

    # --- Single playrec shared by both electrical and acoustic paths ---
    delay_ms, confidence = _playrec_and_correlate(
        output, click, sample_rate, input_idx, output_idx, input_channel,
    )

    # If the correlation failed entirely (playrec error, short rec) the
    # result is (0.0, 0.0) — produce a useful error signal.
    if confidence == 0.0 and delay_ms == 0.0:
        return LatencyResult(
            delay_ms=0.0,
            confidence=0.0,
            accepted=False,
            method="acoustic",
        )

    # Step 1: electrical loopback (confidence >= 0.8).
    if confidence >= 0.8 and delay_ms < record_duration_ms:
        return LatencyResult(
            delay_ms=delay_ms,
            confidence=confidence,
            accepted=True,
            method="electrical_loopback",
        )

    # Step 2: acoustic fallback (confidence >= 0.6).
    accepted = confidence >= 0.6 and delay_ms < record_duration_ms
    return LatencyResult(
        delay_ms=delay_ms,
        confidence=confidence,
        accepted=accepted,
        method="acoustic",
    )
