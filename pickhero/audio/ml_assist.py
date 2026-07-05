"""Optional ML-based pitch/onset assist for the ExperimentalML profile.

This module is an optional scaffold behind a try/except import. It provides
an MLAssist class that wraps an optional ONNX-runtime model (CREPE-small or
similar) for pitch estimation, run in the worker thread from PitchEngine.

Hard rules (enforced in code):
- ml_assist is never imported at module top level — only inside
  PitchEngine.__init__ when profile == "experimental_ml".
- The audio callback never calls into ml_assist.
- If onnxruntime import fails, log a warning and continue with the
  signal-processing path. No crash.
- The model file path is configurable (Config.audio.ml_model_path);
  default is ~/.pickhero/models/crepe_small.onnx.

Training or shipping a model is out of scope for this pass; the goal is that
the architecture allows it without breaking the base app.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """A pitch prediction from the ML model."""
    midi_note: int | None
    frequency: float
    confidence: float


class MLAssist:
    """Optional neural pitch/onset assist.

    Wraps an ONNX-runtime model for pitch estimation. If the model file or
    onnxruntime is absent, this class is not instantiated — PitchEngine falls
    back to the signal-processing consensus.

    Never called from the audio callback. Only used in the worker thread.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._session = None
        self._model = None
        self._available = False

        if model_path is None or not os.path.exists(model_path):
            logger.info("MLAssist: model file not found at %s, using fallback", model_path)
            return

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(model_path)
            self._available = True
            logger.info("MLAssist: loaded model from %s", model_path)
        except ImportError:
            logger.warning(
                "MLAssist: onnxruntime not installed, using signal-processing fallback"
            )
        except Exception as e:
            logger.warning("MLAssist: failed to load model: %s, using fallback", e)

    @property
    def available(self) -> bool:
        """True if a model is loaded and ready to assist."""
        return self._available

    def predict_pitch(self, audio: np.ndarray, sample_rate: int) -> MLPrediction | None:
        """Run the model on a hop-sized audio chunk.

        Returns None if the model is not available or the prediction fails.
        The caller (PitchEngine) falls back to the signal-processing consensus.
        """
        if not self._available or self._session is None:
            return None

        try:
            # Placeholder for actual model inference.
            # A real CREPE-small model would:
            # 1. Resample audio to the model's expected sample rate (16 kHz)
            # 2. Normalize to [-1, 1]
            # 3. Run inference to get a pitch probability distribution
            # 4. Convert the argmax bin to frequency
            # For now, return None to signal fallback.
            return None
        except Exception as e:
            logger.warning("MLAssist: prediction failed: %s, using fallback", e)
            return None


def is_ml_available() -> bool:
    """Check if onnxruntime is installed (without importing it at module level)."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False
