"""Tests for pickhero.audio.latency_calibrator — round-trip latency measurement."""

import numpy as np
import pytest

from pickhero.audio.latency_calibrator import (
    LatencyResult,
    measure_roundtrip_latency,
)


class TestLatencyResult:
    """LatencyResult dataclass and serialization."""

    def test_to_dict_includes_required_keys(self):
        """to_dict() returns all required keys."""
        result = LatencyResult(
            delay_ms=12.5,
            confidence=0.85,
            accepted=True,
            method="electrical_loopback",
        )
        d = result.to_dict()
        assert "delay_ms" in d
        assert "confidence" in d
        assert "accepted" in d
        assert "method" in d
        assert d["delay_ms"] == 12.5
        assert d["confidence"] == 0.85
        assert d["accepted"] is True

    def test_to_dict_values_are_plain_types(self):
        """to_dict() returns plain Python types, not numpy or custom objects."""
        result = LatencyResult(
            delay_ms=5.0,
            confidence=0.9,
            accepted=True,
            method="acoustic",
        )
        d = result.to_dict()
        assert isinstance(d["delay_ms"], float)
        assert isinstance(d["confidence"], float)
        assert isinstance(d["accepted"], bool)
        assert isinstance(d["method"], str)

    def test_fields_accessible(self):
        """All fields are directly accessible."""
        result = LatencyResult(
            delay_ms=8.2,
            confidence=0.75,
            accepted=True,
            method="acoustic",
        )
        assert result.delay_ms == 8.2
        assert result.confidence == 0.75
        assert result.accepted is True
        assert result.method == "acoustic"


class TestMeasureRoundtripLatencyEdgeCases:
    """measure_roundtrip_latency handles edge cases gracefully."""

    def test_no_output_device_returns_accepted_false(self):
        """When no output device is available, returns accepted=False."""
        import sounddevice as sd

        original_devices = sd.query_devices
        try:
            # Make query_devices raise -> _resolve_output_device catches
            # the exception and returns None, triggering the no_output path.
            def failing_query(device=None):
                raise RuntimeError("no audio device available")

            sd.query_devices = failing_query

            result = measure_roundtrip_latency(
                sample_rate=48000,
                input_device=None,
                input_channel=0,
            )

            assert result.accepted is False
            assert result.delay_ms == 0.0
            assert result.confidence == 0.0
            assert result.method == "no_output"
        finally:
            sd.query_devices = original_devices

    def test_playrec_error_returns_graceful_result(self):
        """When playrec fails, returns accepted=False rather than raising."""
        import sounddevice as sd

        original_devices = sd.query_devices
        original_playrec = sd.playrec

        try:
            # _resolve_output_device calls sd.query_devices(idx) which
            # must succeed. _resolve_input_device with input_device=None
            # uses sd.default.device[0] (no query_devices call).
            def fake_devices(device=None):
                if device is not None:
                    return {"name": "fake", "max_input_channels": 1}
                return [
                    {"name": "fake_input", "max_input_channels": 1, "hostapi": 0},
                    {"name": "fake_output", "max_input_channels": 0, "hostapi": 0},
                ]

            def fake_playrec(*args, **kwargs):
                raise RuntimeError("simulated audio error")

            sd.query_devices = fake_devices
            sd.playrec = fake_playrec

            result = measure_roundtrip_latency(
                sample_rate=48000,
                input_device=None,
                input_channel=0,
            )

            assert result.accepted is False
            assert result.delay_ms == 0.0
            assert result.confidence == 0.0
        finally:
            sd.query_devices = original_devices
            sd.playrec = original_playrec

    def test_confidence_below_threshold_not_accepted(self):
        """When cross-correlation confidence is below 0.6, accepted=False."""
        import sounddevice as sd

        original_devices = sd.query_devices
        original_playrec = sd.playrec
        original_wait = sd.wait

        try:
            def fake_devices(device=None):
                if device is not None:
                    return {"name": "fake", "max_input_channels": 1}
                return [
                    {"name": "fake_input", "max_input_channels": 1, "hostapi": 0},
                    {"name": "fake_output", "max_input_channels": 0, "hostapi": 0},
                ]

            recorded_samples = 48000 * 6 // 10

            def fake_playrec(output, samplerate=None, channels=None, device=None, dtype=None):
                return np.zeros((recorded_samples, 1), dtype=np.float32)

            def fake_wait():
                pass

            sd.query_devices = fake_devices
            sd.playrec = fake_playrec
            sd.wait = fake_wait

            result = measure_roundtrip_latency(
                sample_rate=48000,
                input_device=None,
                input_channel=0,
            )

            assert result.accepted is False
            assert "acoustic" in result.method
        finally:
            sd.query_devices = original_devices
            sd.playrec = original_playrec
            sd.wait = original_wait
