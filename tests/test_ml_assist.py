"""Tests for pickhero.audio.ml_assist.

MLAssist wraps an optional ONNX-runtime model. All tests avoid real model
files and onnxruntime being actually installed — they test the fallback
and graceful-degradation contracts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import importlib
import sys

from pickhero.audio.ml_assist import MLAssist, MLPrediction, is_ml_available


SILENT_AUDIO = np.zeros(4096, dtype=np.float32)


class TestMLPredictionDataclass:
    """MLPrediction is a frozen-ish dataclass with known fields."""

    def test_fields(self) -> None:
        pred = MLPrediction(midi_note=60, frequency=261.63, confidence=0.95)
        assert pred.midi_note == 60
        assert pred.frequency == pytest.approx(261.63)
        assert pred.confidence == pytest.approx(0.95)

    def test_midi_none(self) -> None:
        """midi_note is intentionally nullable (no-pitch case)."""
        pred = MLPrediction(midi_note=None, frequency=0.0, confidence=0.0)
        assert pred.midi_note is None
        assert pred.frequency == 0.0
        assert pred.confidence == 0.0

    def test_is_dataclass(self) -> None:
        import dataclasses
        assert dataclasses.is_dataclass(MLPrediction)

    def test_immutable_by_convention(self) -> None:
        """No prescribed frozen=True, but test that fields are set correctly."""
        pred = MLPrediction(midi_note=69, frequency=440.0, confidence=0.9)
        assert pred.midi_note == 69


class TestIsMLAvailable:
    """is_ml_available() reflects whether onnxruntime is importable."""

    def test_returns_bool(self) -> None:
        assert isinstance(is_ml_available(), bool)

    def test_false_when_onnxruntime_missing(self) -> None:
        """Simulate onnxruntime not being installed."""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            from pickhero.audio import ml_assist as ma
            import importlib
            importlib.reload(ma)
            assert ma.is_ml_available() is False

    def test_true_when_onnxruntime_present(self) -> None:
        """Simulate onnxruntime being installed."""
        fake_ort = type("module", (), {"InferenceSession": object})()
        with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
            from pickhero.audio import ml_assist as ma
            import importlib
            importlib.reload(ma)
            assert ma.is_ml_available() is True


class TestMLAssistConstruction:
    """MLAssist.__init__ must be resilient to missing model / missing deps."""

    def test_no_model_path(self) -> None:
        """model_path=None → available() is False, no crash."""
        assist = MLAssist(model_path=None)
        assert assist.available is False
        assert assist._session is None

    def test_nonexistent_model_path(self) -> None:
        """A file path that does not exist → available() is False, no crash."""
        assist = MLAssist(model_path="/nonexistent/path/model.onnx")
        assert assist.available is False

    def test_constructor_accepts_params(self) -> None:
        """Constructor signature accepts model_path and passes it through."""
        assist = MLAssist(model_path="/some/path.onnx")
        assert assist.model_path == "/some/path.onnx"
        # No model file → still graceful
        assert assist.available is False

    def test_model_path_is_none_by_default(self) -> None:
        assist = MLAssist()
        assert assist.model_path is None
        assert assist.available is False


class TestMLAssistPredictPitch:
    """predict_pitch returns None when the model is not available."""

    def test_predict_none_when_not_available(self) -> None:
        assist = MLAssist()
        result = assist.predict_pitch(SILENT_AUDIO, sample_rate=48000)
        assert result is None

    def test_predict_with_numpy_audio(self) -> None:
        """Accept any np.ndarray; return None gracefully when no model."""
        audio = np.random.randn(2048).astype(np.float32)
        assist = MLAssist()
        result = assist.predict_pitch(audio, sample_rate=44100)
        assert result is None

    def test_predict_empty_audio(self) -> None:
        assist = MLAssist()
        result = assist.predict_pitch(np.array([], dtype=np.float32), 48000)
        assert result is None


class TestMLAssistWithMockedOnnx:
    """When onnxruntime *is* present and a model path exists."""

    def test_available_with_mock_model(self, tmp_path: Path) -> None:
        """If onnxruntime can load the model, available() returns True."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake-onnx-model")
        fake_session = object()

        with patch.dict("sys.modules", {"onnxruntime": _make_fake_ort(fake_session)}):
            from pickhero.audio import ml_assist as ma
            importlib.reload(ma)
            assist = ma.MLAssist(model_path=str(model_file))
            assert assist.available is True
            assert assist._session is not None

    def test_prediction_graceful_when_mocked(self, tmp_path: Path) -> None:
        """Predict returns None (placeholder body) even with a loaded model."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake-onnx-model")

        with patch.dict("sys.modules", {"onnxruntime": _make_fake_ort(object())}):
            from pickhero.audio import ml_assist as ma
            importlib.reload(ma)
            assist = ma.MLAssist(model_path=str(model_file))
            # The real predict_pitch body still returns None (placeholder)
            result = assist.predict_pitch(SILENT_AUDIO, 48000)
            assert result is None

class TestMLAssistWithFakeModelFile:
    """A real on-disk file that is not a valid ONNX model."""

    def test_invalid_model_file_caught(self, tmp_path: Path) -> None:
        """A file that exists but onnxruntime can't load → graceful fallback."""
        model_file = tmp_path / "invalid.onnx"
        model_file.write_bytes(b"not-a-real-onnx-model")

        with patch.dict("sys.modules", {"onnxruntime": _make_fake_ort(RuntimeError("bad model"))}):
            from pickhero.audio import ml_assist as ma
            importlib.reload(ma)
            assist = ma.MLAssist(model_path=str(model_file))
            assert assist.available is False

    def test_empty_file(self, tmp_path: Path) -> None:
        model_file = tmp_path / "empty.onnx"
        model_file.write_bytes(b"")

        with patch.dict("sys.modules", {"onnxruntime": _make_fake_ort(RuntimeError("empty file"))}):
            from pickhero.audio import ml_assist as ma
            importlib.reload(ma)
            assist = ma.MLAssist(model_path=str(model_file))
            assert assist.available is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_ort(session_value: object) -> type:
    """Build a fake ``onnxruntime`` module for testing.

    ``InferenceSession(path)`` returns *session_value* if it is not an
    ``Exception``; otherwise it raises.
    """

    class FakeInferenceSession:
        def __init__(self, path: str) -> None:
            if isinstance(session_value, Exception):
                raise session_value
            self._inner = session_value

    return type("onnxruntime", (), {"InferenceSession": FakeInferenceSession})

