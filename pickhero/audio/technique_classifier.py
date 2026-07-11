"""Guitar technique classification scaffold.

Inputs:
  - log-mel patches from the recorded take
  - CQT (constant-Q transform)
  - pitch contour from CREPE/Basic Pitch
  - onset envelope
  - spectral features (centroid, flatness, HNR)
  - tab context (expected technique, string, fret)

Outputs:
  - TechniqueVerdict per note (normal, palm_mute, dead_note, hammer_on,
    pull_off, slide, bend, vibrato, harmonic, pinch_harmonic, tap, scrape)

Training data: GOAT dataset (arXiv:2509.22655) -- paired guitar
audio/tab with technique annotations.

This is a scaffold -- the actual model training is a separate effort.
The interface is defined here so the after-take pipeline can use it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TechniqueClass(Enum):
    """Supported technique classification labels."""
    NORMAL = "normal"
    PALM_MUTE = "palm_mute"
    DEAD_NOTE = "dead_note"
    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE = "slide"
    BEND = "bend"
    VIBRATO = "vibrato"
    HARMONIC = "harmonic"
    PINCH_HARMONIC = "pinch_harmonic"
    TAP = "tap"
    SCRAPE = "scrape"


@dataclass
class FeatureSet:
    """Input features for technique classification."""
    log_mel: np.ndarray | None = None
    cqt: np.ndarray | None = None
    pitch_contour: np.ndarray | None = None
    onset_envelope: np.ndarray | None = None
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    hnr: float = 0.0
    expected_technique: str | None = None
    string: int | None = None
    fret: int | None = None


@dataclass
class ClassificationResult:
    """Output of technique classification for one note."""
    technique: TechniqueClass
    confidence: float
    features_used: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TechniqueClassifier:
    """Guitar technique classification scaffold.

    This is a scaffold -- the actual model training is a separate effort.
    The interface is defined here so the after-take pipeline can use it.

    When no trained model is available, classification falls back to
    the real-time articulation detector's diagnostic labels.
    """

    SUPPORTED_TECHNIQUES = frozenset(t.value for t in TechniqueClass)

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._model = None
        self._available = False
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a trained model. Scaffold -- no-op until model exists."""
        import os
        if os.path.exists(path):
            # Scaffold: would load ONNX/torch model here
            self._available = False

    @property
    def is_available(self) -> bool:
        """True if a trained model is loaded and ready."""
        return self._available

    def classify(self, features: FeatureSet) -> ClassificationResult:
        """Classify technique from features.

        Without a trained model, falls back to the expected technique
        from tab context, or NORMAL if no expectation.
        """
        if self._available:
            return self._classify_with_model(features)
        return self._fallback_classify(features)

    def _classify_with_model(self, features: FeatureSet) -> ClassificationResult:
        """Run model inference. Scaffold -- not implemented."""
        return self._fallback_classify(features)

    def _fallback_classify(self, features: FeatureSet) -> ClassificationResult:
        """Fallback: use tab context or heuristic features."""
        if features.expected_technique:
            try:
                technique = TechniqueClass(features.expected_technique)
            except ValueError:
                technique = TechniqueClass.NORMAL
        else:
            technique = TechniqueClass.NORMAL

        return ClassificationResult(
            technique=technique,
            confidence=0.5,
            features_used=['expected_technique'],
            metadata={'fallback': True},
        )

    def classify_batch(self, features_list: list[FeatureSet]) -> list[ClassificationResult]:
        """Classify multiple notes. Convenience method."""
        return [self.classify(f) for f in features_list]
