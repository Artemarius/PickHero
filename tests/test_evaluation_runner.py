"""Tests for pickhero/evaluation/runner.py — CorpusEvaluator + EvaluationRun."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pickhero.audio.evidence import (
    ChordVerification,
    NoteVerification,
    PitchEvidence,
    TechniqueVerification,
)
from pickhero.audio.match_mode import MatchMode
from pickhero.evaluation.audio import AudioHealth, AudioWindow
from pickhero.evaluation.manifest import CorpusCase, CorpusExpectedNote, CorpusSplit, EventKind
from pickhero.evaluation.runner import CorpusEvaluator, EvaluationConfig, EvaluationRun


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_case(
    *,
    case_id: str = "test-001",
    event_kind: str = "single_note",
    expected_present: bool = True,
    midi: int = 40,
    technique: str | None = None,
    technique_present: bool | None = None,
    start_s: float = 0.1,
    end_s: float = 1.0,
    audio_path: str = "/fake/test.wav",
    negative_reason: str | None = None,
) -> CorpusCase:
    notes: tuple[CorpusExpectedNote, ...]
    if event_kind == "silence":
        notes = ()
        expected_present = False
        negative_reason = negative_reason or "silence"
    else:
        notes = (CorpusExpectedNote(midi=midi),)
    return CorpusCase(
        case_id=case_id,
        audio_path=audio_path,
        source="tests",
        split=CorpusSplit.TEST,
        event_kind=EventKind(event_kind),
        start_s=start_s,
        end_s=end_s,
        expected_present=expected_present,
        notes=notes,
        technique=technique,
        technique_present=technique_present,
        negative_reason=negative_reason or ("negative" if not expected_present else None),
        window_before_ms=50.0,
        window_after_ms=200.0,
    )


def _health(peak: float = -6.0) -> AudioHealth:
    return AudioHealth(
        peak_dbfs=peak, rms_dbfs=-18.0, dc_offset=0.001, clipped_fraction=0.0
    )


def _mock_window(samples: np.ndarray | None = None) -> AudioWindow:
    if samples is None:
        samples = np.zeros(4800, dtype=np.float32)
    return AudioWindow(
        samples=samples,
        sample_rate=48000,
        expected_onset_offset_ms=50.0,
        health=_health(),
    )


def _note_verification(
    present: bool = True, confidence: float = 0.95, cents: float | None = 2.0
) -> NoteVerification:
    return NoteVerification(
        is_pitch_present=present,
        is_onset_present=True,
        pitch_evidence=(
            PitchEvidence(midi_note=40, cents_error=cents, confidence=confidence, source="test")
        ),
        onset_ms=55.0,
        harmonic_score=0.9,
        timing_error_ms=5.0,
        alias_risk=0.1,
    )


def _chord_verification(present_count: int = 3) -> ChordVerification:
    notes = [_note_verification(present=True) for _ in range(present_count)]
    return ChordVerification(
        notes=notes,
        partial=False,
        total_harmonic_energy=3.0,
        observed_pitch_classes=frozenset({4, 7, 0}),
        pitch_class_energy={0: 0.9, 4: 0.85, 7: 0.8},
        quality_score=0.85,
    )


def _technique_verification() -> TechniqueVerification:
    return TechniqueVerification(
        technique="vibrato",
        is_present=True,
        confidence=0.8,
        uncertain=False,
        quality=0.75,
    )


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    def test_defaults(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.mode == MatchMode.JUDGE
        assert cfg.sample_rate == 48000
        assert cfg.onset_tolerance_ms == 45.0
        assert cfg.silence_threshold_db == -55.0
        assert cfg.require_onset_for_event is False

    def test_custom_values(self) -> None:
        cfg = EvaluationConfig(
            mode=MatchMode.PRACTICE,
            sample_rate=44100,
            onset_tolerance_ms=60.0,
            silence_threshold_db=-50.0,
            require_onset_for_event=True,
        )
        assert cfg.mode == MatchMode.PRACTICE
        assert cfg.sample_rate == 44100
        assert cfg.onset_tolerance_ms == 60.0
        assert cfg.silence_threshold_db == -50.0
        assert cfg.require_onset_for_event is True


# ---------------------------------------------------------------------------
# CorpusEvaluator — constructor
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorConstructor:
    def test_accepts_config(self) -> None:
        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg)
        assert evaluator.config is cfg
        assert evaluator.verifier is not None
        assert evaluator.audio is not None
        assert evaluator.policy is not None

    def test_injects_dependencies(self) -> None:
        verifier = MagicMock()
        audio = MagicMock()
        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        assert evaluator.verifier is verifier
        assert evaluator.audio is audio

    def test_mode_maps_to_policy(self) -> None:
        cfg = EvaluationConfig(mode=MatchMode.ARCADE)
        evaluator = CorpusEvaluator(cfg)
        assert evaluator.policy.name == "arcade"

        cfg2 = EvaluationConfig(mode=MatchMode.JUDGE)
        evaluator2 = CorpusEvaluator(cfg2)
        assert evaluator2.policy.name == "judge"


# ---------------------------------------------------------------------------
# CorpusEvaluator — empty corpus
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorEmpty:
    def test_empty_corpus_returns_empty_metrics(self) -> None:
        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([], manifest_path)

        assert run.records == []
        assert isinstance(run.summary, dict)
        assert run.summary["case_count"] == 0
        assert run.summary["overall"]["event"]["precision"] == 0.0
        assert run.summary["overall"]["event"]["recall"] == 0.0


# ---------------------------------------------------------------------------
# CorpusEvaluator — single note cases
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorSingleNote:
    def test_expected_present_happy_path(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(
            present=True, confidence=0.92
        )
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="n001", midi=40, expected_present=True)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        assert len(run.records) == 1
        rec = run.records[0]
        assert rec.case_id == "n001"
        assert rec.expected_present is True
        assert rec.predicted_present is True
        assert rec.score == 0.92
        assert rec.failure_reasons == ()
        assert rec.cents_error == 2.0
        assert rec.alias_risk == 0.1

    def test_expected_present_but_detector_misses(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(
            present=False, confidence=0.3
        )
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="n002", midi=40, expected_present=True)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert rec.predicted_present is False
        assert "false_reject" in rec.failure_reasons

    def test_expected_absent_but_detector_says_present(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(
            present=True, confidence=0.95
        )
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(
            case_id="n003", midi=40, expected_present=False, negative_reason="noise"
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert rec.expected_present is False
        assert rec.predicted_present is True
        assert "false_accept" in rec.failure_reasons

    def test_high_alias_risk_failure(self) -> None:
        nv = _note_verification(present=True, confidence=0.8)
        nv.alias_risk = 0.9
        verifier = MagicMock()
        verifier.verify_single_note.return_value = nv
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="n004", midi=40, expected_present=True)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert "high_alias_risk" in rec.failure_reasons

    def test_require_onset_for_event_predicted_becomes_false(self) -> None:
        verifier = MagicMock()
        nv = _note_verification(present=True, confidence=0.9)
        # is_onset_present is False
        nv.is_onset_present = False
        verifier.verify_single_note.return_value = nv

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig(require_onset_for_event=True)
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="n005", midi=40, expected_present=True)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        # pitch is true but onset is false → predicted should be false
        assert rec.predicted_present is False
        assert "false_reject" in rec.failure_reasons
# ---------------------------------------------------------------------------
# CorpusEvaluator — silence cases
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorSilence:
    def test_silence_detected(self) -> None:
        verifier = MagicMock()
        verifier.verify_silence.return_value = True
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(
            case_id="s001", event_kind="silence", negative_reason="silence"
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert rec.predicted_present is False
        assert rec.event_kind == "silence"
        assert rec.failure_reasons == ()

    def test_silence_not_detected(self) -> None:
        verifier = MagicMock()
        verifier.verify_silence.return_value = False
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(
            case_id="s002", event_kind="silence", negative_reason="silence"
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert "unexpected_signal" in rec.failure_reasons

    def test_silence_score_from_rms(self) -> None:
        verifier = MagicMock()
        verifier.verify_silence.return_value = True
        samples = np.ones(4800, dtype=np.float32) * 0.01
        audio_health = AudioHealth(
            peak_dbfs=-40.0, rms_dbfs=-40.0, dc_offset=0.0, clipped_fraction=0.0
        )
        audio = MagicMock()
        audio.window_for_case.return_value = AudioWindow(
            samples=samples,
            sample_rate=48000,
            expected_onset_offset_ms=None,
            health=audio_health,
        )
        case = _make_case(
            case_id="s003", event_kind="silence", negative_reason="silence"
        )

        cfg = EvaluationConfig(silence_threshold_db=-55.0)
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        # score = max(0.0, min(1.0, (rms_dbfs - threshold) / 30.0))
        # (-40.0 - (-55.0)) / 30.0 = 15.0 / 30.0 = 0.5
        assert rec.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CorpusEvaluator — chord cases
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorChord:
    def test_chord_hit(self) -> None:
        verifier = MagicMock()
        cv = _chord_verification(present_count=3)
        # Observed pitch classes match expected C major (C, E, G)
        cv.observed_pitch_classes = frozenset({0, 4, 7})
        cv.pitch_class_energy = {0: 0.9, 4: 0.85, 7: 0.8}
        verifier.verify_chord.return_value = cv

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        notes = (
            CorpusExpectedNote(midi=48),  # C3 → pc 0
            CorpusExpectedNote(midi=52),  # E3 → pc 4
            CorpusExpectedNote(midi=55),  # G3 → pc 7
        )
        case = CorpusCase(
            case_id="c001",
            audio_path="/fake/test.wav",
            source="tests",
            split=CorpusSplit.TEST,
            event_kind=EventKind.CHORD,
            start_s=0.1,
            end_s=1.0,
            expected_present=True,
            notes=notes,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert rec.predicted_present is True
        assert rec.chord_verdict is not None
        assert rec.failure_reasons == ()

    def test_chord_miss(self) -> None:
        verifier = MagicMock()
        # Return verification where all notes are absent
        cv = _chord_verification(present_count=3)
        for note in cv.notes:
            note.is_pitch_present = False
        verifier.verify_chord.return_value = cv

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        notes = (
            CorpusExpectedNote(midi=40),
            CorpusExpectedNote(midi=44),
            CorpusExpectedNote(midi=47),
        )
        case = CorpusCase(
            case_id="c002",
            audio_path="/fake/test.wav",
            source="tests",
            split=CorpusSplit.TEST,
            event_kind=EventKind.CHORD,
            start_s=0.1,
            end_s=1.0,
            expected_present=True,
            notes=notes,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert rec.predicted_present is False
        assert "false_reject" in rec.failure_reasons

    def test_chord_with_missing_critical_roles(self) -> None:
        verifier = MagicMock()
        cv = _chord_verification(present_count=3)
        # Set observed to only root+fifth, missing the third
        cv.observed_pitch_classes = frozenset({0, 7})
        cv.pitch_class_energy = {0: 0.9, 7: 0.8}
        verifier.verify_chord.return_value = cv

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        notes = (
            CorpusExpectedNote(midi=40),  # E2 → pc 4
            CorpusExpectedNote(midi=44),  # G#2 → pc 8
            CorpusExpectedNote(midi=47),  # B2 → pc 11
        )
        case = CorpusCase(
            case_id="c003",
            audio_path="/fake/test.wav",
            source="tests",
            split=CorpusSplit.TEST,
            event_kind=EventKind.CHORD,
            start_s=0.1,
            end_s=1.0,
            expected_present=True,
            notes=notes,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        # score_chord infers root from expected pitch classes
        assert rec.predicted_present is False


# ---------------------------------------------------------------------------
# CorpusEvaluator — technique cases
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorTechnique:
    def test_technique_correct(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(
            present=True, confidence=0.9
        )
        verifier.verify_technique.return_value = _technique_verification()

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(
            case_id="t001",
            midi=40,
            technique="vibrato",
            technique_present=True,
            event_kind="technique",
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert rec.predicted_present is True
        assert rec.technique == "vibrato"
        assert rec.technique_detected is True
        assert rec.technique_uncertain is False
        assert rec.failure_reasons == ()

    def test_technique_mismatch(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(
            present=True, confidence=0.9
        )
        tv = _technique_verification()
        tv.is_present = False
        verifier.verify_technique.return_value = tv

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(
            case_id="t002",
            midi=40,
            technique="vibrato",
            technique_present=True,
            event_kind="technique",
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert "technique_false_reject" in rec.failure_reasons

    def test_technique_uncertain_skips_failure(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(
            present=True, confidence=0.9
        )
        tv = _technique_verification()
        tv.is_present = False
        tv.uncertain = True
        verifier.verify_technique.return_value = tv

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(
            case_id="t003",
            midi=40,
            technique="vibrato",
            technique_present=True,
            event_kind="technique",
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        # uncertain=True means no technique failure appended
        assert "technique_false_reject" not in rec.failure_reasons


# ---------------------------------------------------------------------------
# CorpusEvaluator — audio health failures
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorAudioHealth:
    def test_clipped_audio_adds_failure(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = AudioWindow(
            samples=np.zeros(4800, dtype=np.float32),
            sample_rate=48000,
            expected_onset_offset_ms=50.0,
            health=AudioHealth(
                peak_dbfs=0.0,
                rms_dbfs=-6.0,
                dc_offset=0.0,
                clipped_fraction=0.15,
            ),
        )

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="h001", midi=40)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert "audio_clipped" in rec.failure_reasons

    def test_dc_offset_adds_failure(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = AudioWindow(
            samples=np.zeros(4800, dtype=np.float32),
            sample_rate=48000,
            expected_onset_offset_ms=50.0,
            health=AudioHealth(
                peak_dbfs=-6.0,
                rms_dbfs=-18.0,
                dc_offset=0.05,
                clipped_fraction=0.0,
            ),
        )

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="h002", midi=40)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        assert "dc_offset" in rec.failure_reasons


# ---------------------------------------------------------------------------
# CorpusEvaluator — multiple cases, summary
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorSummary:
    def test_mixed_cases_produce_metrics(self) -> None:
        verifier = MagicMock()

        def _fake_verify_single_note(audio, midi, mode, **kw):
            if midi == 40:
                return _note_verification(present=True, confidence=0.9)
            return _note_verification(present=False, confidence=0.2)

        verifier.verify_single_note.side_effect = _fake_verify_single_note
        verifier.verify_silence.return_value = True

        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)

        cases = [
            _make_case(case_id="tp1", midi=40, expected_present=True),
            _make_case(case_id="fn1", midi=45, expected_present=True),
            _make_case(case_id="sil1", event_kind="silence", negative_reason="silence"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate(cases, manifest_path)

        assert len(run.records) == 3

        # tp1: present + predicted present → TP
        # fn1: present + predicted absent → FN
        # sil1: silence + silence detected → TN (expected_present=False, predicted_present=False)
        assert run.records[0].failure_reasons == ()
        assert "false_reject" in run.records[1].failure_reasons
        assert run.records[2].failure_reasons == ()

        summary = run.summary
        assert summary["case_count"] == 3
        assert summary["mode"] == "judge"
        overall = summary["overall"]
        assert isinstance(overall, dict)
        event = overall["event"]
        assert isinstance(event, dict)

        # 1 TP (tp1), 0 FP, 1 FN (fn1), 1 TN (sil1)
        assert event["tp"] == 1
        assert event["fp"] == 0
        assert event["fn"] == 1
        assert event["tn"] == 1

# ---------------------------------------------------------------------------
# EvaluationRun — output format
# ---------------------------------------------------------------------------


class TestEvaluationRunOutput:
    def test_write_outputs_files(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="fmt-001", midi=40)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

            output_dir = Path(tmp) / "output"
            run.write(output_dir)

            assert (output_dir / "records.jsonl").exists()
            assert (output_dir / "summary.json").exists()
            assert (output_dir / "failures.jsonl").exists()
            assert (output_dir / "report.md").exists()

    def test_records_jsonl_content(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="fmt-002", midi=40)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

            output_dir = Path(tmp) / "output"
            run.write(output_dir)

            lines = (output_dir / "records.jsonl").read_text().strip().splitlines()
            assert len(lines) == 1
            payload = json.loads(lines[0])
            assert payload["case_id"] == "fmt-002"
            assert payload["expected_present"] is True
            assert payload["predicted_present"] is True
            assert payload["score"] == 0.95  # from _note_verification confidence

    def test_summary_json_content(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig(mode=MatchMode.PRACTICE, sample_rate=44100)
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="fmt-003", midi=40, expected_present=True)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

            output_dir = Path(tmp) / "output"
            run.write(output_dir)

            summary = json.loads((output_dir / "summary.json").read_text())
            assert summary["mode"] == "practice"
            assert summary["sample_rate"] == 44100
            assert summary["case_count"] == 1
            assert "run" in summary
            assert summary["run"]["config"]["mode"] == "practice"

    def test_failures_jsonl_only_failures(self) -> None:
        verifier = MagicMock()

        def _side_effect(audio, midi, mode, **kw):
            if midi == 40:
                return _note_verification(present=True)
            return _note_verification(present=False)

        verifier.verify_single_note.side_effect = _side_effect
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        cases = [
            _make_case(case_id="ok", midi=40, expected_present=True),
            _make_case(case_id="fail", midi=45, expected_present=True),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate(cases, manifest_path)

            output_dir = Path(tmp) / "output"
            run.write(output_dir)

            failures_text = (output_dir / "failures.jsonl").read_text().strip()
            assert failures_text != ""
            failure_ids = []
            for line in failures_text.splitlines():
                if line:
                    failure_ids.append(json.loads(line)["case_id"])
            assert failure_ids == ["fail"]

    def test_report_md_generated(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="rpt-001", midi=40)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

            output_dir = Path(tmp) / "output"
            run.write(output_dir)

            report = (output_dir / "report.md").read_text()
            assert "# PickHero corpus evaluation" in report
            assert "Cases: **1**" in report

    def test_to_dict_includes_all_keys(self) -> None:
        verifier = MagicMock()
        verifier.verify_single_note.return_value = _note_verification(present=True)
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)
        case = _make_case(case_id="dict-001", midi=40)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        data = rec.to_dict()
        assert isinstance(data, dict)
        assert data["case_id"] == "dict-001"
        assert data["score"] == 0.95
        assert data["failure_reasons"] == []


# ---------------------------------------------------------------------------
# CorpusEvaluator — integration with real score_chord (chord scoring logic)
# ---------------------------------------------------------------------------


class TestCorpusEvaluatorChordScoreIntegration:
    """Verify that real score_chord logic is exercised when ChordVerification is realistic."""

    def test_score_chord_hit_verdict(self) -> None:
        verifier = MagicMock()
        cv = ChordVerification(
            notes=[
                NoteVerification(
                    is_pitch_present=True,
                    is_onset_present=True,
                    pitch_evidence=PitchEvidence(
                        midi_note=40, cents_error=1.0, confidence=0.95, source="test"
                    ),
                    onset_ms=52.0,
                    harmonic_score=0.95,
                    timing_error_ms=2.0,
                    alias_risk=0.05,
                ),
                NoteVerification(
                    is_pitch_present=True,
                    is_onset_present=True,
                    pitch_evidence=PitchEvidence(
                        midi_note=44, cents_error=-0.5, confidence=0.88, source="test"
                    ),
                    onset_ms=53.0,
                    harmonic_score=0.85,
                    timing_error_ms=3.0,
                    alias_risk=0.1,
                ),
                NoteVerification(
                    is_pitch_present=True,
                    is_onset_present=True,
                    pitch_evidence=PitchEvidence(
                        midi_note=47, cents_error=0.3, confidence=0.92, source="test"
                    ),
                    onset_ms=51.0,
                    harmonic_score=0.9,
                    timing_error_ms=1.0,
                    alias_risk=0.08,
                ),
            ],
            partial=False,
            total_harmonic_energy=2.85,
            observed_pitch_classes=frozenset({4, 7, 11}),
            pitch_class_energy={0: 0.0, 4: 0.9, 7: 0.85, 11: 0.88},
            quality_score=0.9,
        )
        verifier.verify_chord.return_value = cv
        audio = MagicMock()
        audio.window_for_case.return_value = _mock_window()

        cfg = EvaluationConfig()
        evaluator = CorpusEvaluator(cfg, verifier=verifier, audio=audio)

        notes = (
            CorpusExpectedNote(midi=40),  # E2 → pc 4
            CorpusExpectedNote(midi=44),  # G#2 → pc 8
            CorpusExpectedNote(midi=47),  # B2 → pc 11
        )
        case = CorpusCase(
            case_id="integ-c001",
            audio_path="/fake/test.wav",
            source="tests",
            split=CorpusSplit.TEST,
            event_kind=EventKind.CHORD,
            start_s=0.1,
            end_s=1.0,
            expected_present=True,
            notes=notes,
        )

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            manifest_path.write_text("")
            run = evaluator.evaluate([case], manifest_path)

        rec = run.records[0]
        # This should be a miss because expected pitch classes (4, 8, 11) don't
        # match observed (4, 7, 11) — the G#2's pc=8 is missing, pc=7 is observed.
        assert rec.predicted_present is False
        assert rec.chord_verdict is not None
