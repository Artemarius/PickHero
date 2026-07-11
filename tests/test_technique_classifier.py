"""Tests for pickhero.audio.technique_classifier -- technique scaffold."""

import pytest

from pickhero.audio.technique_classifier import (
    ClassificationResult,
    FeatureSet,
    TechniqueClass,
    TechniqueClassifier,
)


class TestTechniqueClass:
    """TechniqueClass enum must cover all 12 supported labels."""

    def test_all_values_present(self):
        """There are exactly 12 technique classes."""
        values = list(TechniqueClass)
        assert len(values) == 12

    def test_normal_member(self):
        assert TechniqueClass.NORMAL.value == "normal"

    def test_palm_mute_member(self):
        assert TechniqueClass.PALM_MUTE.value == "palm_mute"

    def test_dead_note_member(self):
        assert TechniqueClass.DEAD_NOTE.value == "dead_note"

    def test_hammer_on_member(self):
        assert TechniqueClass.HAMMER_ON.value == "hammer_on"

    def test_pull_off_member(self):
        assert TechniqueClass.PULL_OFF.value == "pull_off"

    def test_slide_member(self):
        assert TechniqueClass.SLIDE.value == "slide"

    def test_bend_member(self):
        assert TechniqueClass.BEND.value == "bend"

    def test_vibrato_member(self):
        assert TechniqueClass.VIBRATO.value == "vibrato"

    def test_harmonic_member(self):
        assert TechniqueClass.HARMONIC.value == "harmonic"

    def test_pinch_harmonic_member(self):
        assert TechniqueClass.PINCH_HARMONIC.value == "pinch_harmonic"

    def test_tap_member(self):
        assert TechniqueClass.TAP.value == "tap"

    def test_scrape_member(self):
        assert TechniqueClass.SCRAPE.value == "scrape"

    def test_supported_techniques_has_all_values(self):
        """The classifier's SUPPORTED_TECHNIQUES matches the enum."""
        assert TechniqueClassifier.SUPPORTED_TECHNIQUES == frozenset(
            t.value for t in TechniqueClass
        )


class TestFeatureSet:
    """FeatureSet dataclass defaults."""

    def test_defaults_are_none_or_zero(self):
        features = FeatureSet()
        assert features.log_mel is None
        assert features.cqt is None
        assert features.pitch_contour is None
        assert features.onset_envelope is None
        assert features.spectral_centroid == 0.0
        assert features.spectral_flatness == 0.0
        assert features.hnr == 0.0
        assert features.expected_technique is None
        assert features.string is None
        assert features.fret is None


class TestClassificationResult:
    """ClassificationResult construction."""

    def test_construct_with_technique_and_confidence(self):
        result = ClassificationResult(
            technique=TechniqueClass.NORMAL, confidence=0.95
        )
        assert result.technique == TechniqueClass.NORMAL
        assert result.confidence == 0.95
        assert result.features_used == []
        assert result.metadata == {}

    def test_with_features_and_metadata(self):
        result = ClassificationResult(
            technique=TechniqueClass.VIBRATO,
            confidence=0.8,
            features_used=['cqt', 'pitch_contour'],
            metadata={'fallback': False, 'model_version': '1.0'},
        )
        assert result.technique == TechniqueClass.VIBRATO
        assert result.confidence == 0.8
        assert 'cqt' in result.features_used
        assert result.metadata['model_version'] == '1.0'


class TestTechniqueClassifier:
    """TechniqueClassifier scaffold behaviour."""

    def test_no_model_defaults_to_unavailable(self):
        classifier = TechniqueClassifier()
        assert classifier.is_available is False

    def test_non_existent_model_path_does_not_load(self):
        classifier = TechniqueClassifier(model_path="/nonexistent/path/model.onnx")
        assert classifier.is_available is False

    def test_classify_with_expected_technique_returns_matching_class(self):
        classifier = TechniqueClassifier()
        features = FeatureSet(expected_technique="hammer_on")
        result = classifier.classify(features)
        assert result.technique == TechniqueClass.HAMMER_ON

    def test_classify_with_unknown_technique_returns_normal(self):
        classifier = TechniqueClassifier()
        features = FeatureSet(expected_technique="unknown_flamenco")
        result = classifier.classify(features)
        assert result.technique == TechniqueClass.NORMAL

    def test_classify_with_no_expected_technique_returns_normal(self):
        classifier = TechniqueClassifier()
        features = FeatureSet()
        result = classifier.classify(features)
        assert result.technique == TechniqueClass.NORMAL

    def test_classify_batch_processes_all_features(self):
        classifier = TechniqueClassifier()
        features_list = [
            FeatureSet(expected_technique="bend"),
            FeatureSet(expected_technique="tap"),
            FeatureSet(),
        ]
        results = classifier.classify_batch(features_list)
        assert len(results) == 3
        assert results[0].technique == TechniqueClass.BEND
        assert results[1].technique == TechniqueClass.TAP
        assert results[2].technique == TechniqueClass.NORMAL

    def test_fallback_confidence_is_0_5(self):
        classifier = TechniqueClassifier()
        features = FeatureSet()
        result = classifier.classify(features)
        assert result.confidence == 0.5

    def test_fallback_metadata_marks_fallback(self):
        classifier = TechniqueClassifier()
        features = FeatureSet()
        result = classifier.classify(features)
        assert result.metadata.get('fallback') is True

    def test_fallback_features_used_list(self):
        classifier = TechniqueClassifier()
        features = FeatureSet(expected_technique="slide")
        result = classifier.classify(features)
        assert 'expected_technique' in result.features_used
