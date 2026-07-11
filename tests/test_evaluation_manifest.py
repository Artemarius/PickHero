"""Tests for pickhero.evaluation.manifest — CorpusCase, CorpusExpectedNote, iter_manifest, etc."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pickhero.evaluation.manifest import (
    CorpusCase,
    CorpusExpectedNote,
    CorpusSplit,
    EventKind,
    iter_manifest,
    load_manifest,
    write_manifest,
)


# ---------------------------------------------------------------------------
# EventKind
# ---------------------------------------------------------------------------


class TestEventKind:
    def test_expected_values(self):
        assert EventKind.SINGLE_NOTE.value == "single_note"
        assert EventKind.CHORD.value == "chord"
        assert EventKind.TECHNIQUE.value == "technique"
        assert EventKind.SILENCE.value == "silence"

    def test_is_str_enum(self):
        assert issubclass(EventKind, str)


# ---------------------------------------------------------------------------
# CorpusSplit
# ---------------------------------------------------------------------------


class TestCorpusSplit:
    def test_expected_values(self):
        assert CorpusSplit.CALIBRATION.value == "calibration"
        assert CorpusSplit.TEST.value == "test"
        assert CorpusSplit.DEVELOPMENT.value == "development"

    def test_is_str_enum(self):
        assert issubclass(CorpusSplit, str)


# ---------------------------------------------------------------------------
# CorpusExpectedNote
# ---------------------------------------------------------------------------


class TestCorpusExpectedNote:
    @staticmethod
    def _minimal_dict() -> dict:
        return {"midi": 40}

    @staticmethod
    def _full_dict() -> dict:
        return {"midi": 42, "string": 5, "fret": 3, "role": "root"}

    def test_from_dict_minimal(self):
        note = CorpusExpectedNote.from_dict(self._minimal_dict())
        assert note.midi == 40
        assert note.string is None
        assert note.fret is None
        assert note.role is None

    def test_from_dict_full(self):
        note = CorpusExpectedNote.from_dict(self._full_dict())
        assert note.midi == 42
        assert note.string == 5
        assert note.fret == 3
        assert note.role == "root"

    def test_from_dict_partial(self):
        note = CorpusExpectedNote.from_dict({"midi": 44, "string": 4})
        assert note.midi == 44
        assert note.string == 4
        assert note.fret is None
        assert note.role is None

    def test_from_dict_preserves_none_optional(self):
        note = CorpusExpectedNote.from_dict(
            {"midi": 45, "string": None, "fret": None, "role": None}
        )
        assert note.midi == 45
        assert note.string is None
        assert note.fret is None
        assert note.role is None

    def test_midi_required(self):
        with pytest.raises(KeyError):
            CorpusExpectedNote.from_dict({})

    def test_frozen_dataclass(self):
        note = CorpusExpectedNote.from_dict(self._full_dict())
        with pytest.raises(AttributeError):
            note.midi = 50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CorpusCase — from_dict
# ---------------------------------------------------------------------------


class TestCorpusCaseFromDict:
    """Tests for CorpusCase.from_dict()."""

    @staticmethod
    def _single_note_dict(**overrides) -> dict:
        base = {
            "case_id": "test-001",
            "audio_path": "/audio/test.wav",
            "source": "test_source",
            "split": "test",
            "event_kind": "single_note",
            "start_s": 0.0,
            "end_s": 2.0,
            "expected_present": True,
            "notes": [{"midi": 40}],
        }
        base.update(overrides)
        return base

    def test_minimal_single_note(self):
        case = CorpusCase.from_dict(self._single_note_dict())
        assert case.case_id == "test-001"
        assert case.audio_path == "/audio/test.wav"
        assert case.source == "test_source"
        assert case.split == CorpusSplit.TEST
        assert case.event_kind == EventKind.SINGLE_NOTE
        assert case.start_s == 0.0
        assert case.end_s == 2.0
        assert case.expected_present is True
        assert len(case.notes) == 1
        assert case.notes[0].midi == 40
        assert case.technique is None
        assert case.technique_present is None
        assert case.expected_onset_s is None
        assert case.annotation_confidence == 1.0
        assert case.negative_reason is None
        assert case.metadata == {}
        assert case.technique_context == {}
        assert case.window_before_ms == 120.0
        assert case.window_after_ms is None

    def test_full_dict(self):
        raw = {
            "case_id": "test-099",
            "audio_path": "~/corpus/notes.wav",
            "source": "manual",
            "split": "development",
            "event_kind": "technique",
            "start_s": 0.5,
            "end_s": 1.5,
            "expected_present": False,
            "notes": [{"midi": 52, "string": 6, "fret": 1, "role": "root"}],
            "technique": "vibrato",
            "technique_present": True,
            "expected_onset_s": 0.55,
            "annotation_confidence": 0.9,
            "negative_reason": "wrong string",
            "metadata": {"tuning": "E2"},
            "technique_context": {"type": "wide"},
            "window_before_ms": 200.0,
            "window_after_ms": 300.0,
        }
        case = CorpusCase.from_dict(raw)
        assert case.case_id == "test-099"
        assert case.audio_path == "~/corpus/notes.wav"
        assert case.source == "manual"
        assert case.split == CorpusSplit.DEVELOPMENT
        assert case.event_kind == EventKind.TECHNIQUE
        assert case.start_s == 0.5
        assert case.end_s == 1.5
        assert case.expected_present is False
        assert case.notes[0].midi == 52
        assert case.notes[0].string == 6
        assert case.notes[0].fret == 1
        assert case.notes[0].role == "root"
        assert case.technique == "vibrato"
        assert case.technique_present is True
        assert case.expected_onset_s == 0.55
        assert case.annotation_confidence == 0.9
        assert case.negative_reason == "wrong string"
        assert case.metadata == {"tuning": "E2"}
        assert case.technique_context == {"type": "wide"}
        assert case.window_before_ms == 200.0
        assert case.window_after_ms == 300.0

    def test_with_ordered_notes(self):
        raw = self._single_note_dict(
            notes=[
                {"midi": 40, "string": 1, "role": "root"},
                {"midi": 44, "string": 2, "role": "third"},
            ],
            event_kind="chord",
        )
        case = CorpusCase.from_dict(raw)
        assert len(case.notes) == 2
        assert case.notes[0].midi == 40
        assert case.notes[0].string == 1
        assert case.notes[1].midi == 44
        assert case.notes[1].role == "third"

    def test_schema_version_mismatch(self):
        raw = self._single_note_dict(schema_version=2)
        with pytest.raises(ValueError, match="unsupported corpus schema_version 2"):
            CorpusCase.from_dict(raw)

    def test_notes_must_be_array(self):
        raw = self._single_note_dict(notes="invalid")
        with pytest.raises(ValueError, match="notes must be an array"):
            CorpusCase.from_dict(raw)

    def test_metadata_must_be_dict(self):
        raw = self._single_note_dict(metadata="strnotdict")
        with pytest.raises(ValueError, match="metadata.*must be object"):
            CorpusCase.from_dict(raw)

    def test_technique_context_must_be_dict(self):
        raw = self._single_note_dict(technique_context="strnotdict")
        with pytest.raises(ValueError, match="technique_context.*must be object"):
            CorpusCase.from_dict(raw)

    def test_missing_audio_path_raises(self):
        raw = self._single_note_dict()
        del raw["audio_path"]
        with pytest.raises(KeyError):
            CorpusCase.from_dict(raw)

    def test_missing_case_id_raises(self):
        raw = self._single_note_dict()
        del raw["case_id"]
        with pytest.raises(KeyError):
            CorpusCase.from_dict(raw)

    def test_default_split_is_test(self):
        raw = self._single_note_dict()
        del raw["split"]
        case = CorpusCase.from_dict(raw)
        assert case.split == CorpusSplit.TEST

    def test_default_expected_present_is_true(self):
        raw = self._single_note_dict()
        del raw["expected_present"]
        case = CorpusCase.from_dict(raw)
        assert case.expected_present is True

    def test_default_annotation_confidence_is_1(self):
        """When annotation_confidence is omitted, defaults to 1.0."""
        raw = self._single_note_dict()
        # Ensure annotation_confidence is not in the dict
        raw.pop("annotation_confidence", None)
        case = CorpusCase.from_dict(raw)
        assert case.annotation_confidence == 1.0

    def test_default_window_before_ms(self):
        """When window_before_ms is omitted, defaults to 120.0."""
        raw = self._single_note_dict()
        raw.pop("window_before_ms", None)
        case = CorpusCase.from_dict(raw)
        assert case.window_before_ms == 120.0

    def test_window_after_ms_optional(self):
        """When window_after_ms is omitted, defaults to None."""
        raw = self._single_note_dict()
        raw.pop("window_after_ms", None)
        case = CorpusCase.from_dict(raw)
        assert case.window_after_ms is None


# ---------------------------------------------------------------------------
# CorpusCase — to_dict
# ---------------------------------------------------------------------------


class TestCorpusCaseToDict:
    @staticmethod
    def _single_note_case(**overrides) -> CorpusCase:
        kwargs = dict(
            case_id="to-dict-001",
            audio_path="/audio/to_dict.wav",
            source="test_source",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.5,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40, string=1),),
        )
        kwargs.update(overrides)
        return CorpusCase(**kwargs)

    def test_includes_schema_version(self):
        d = self._single_note_case().to_dict()
        assert d["schema_version"] == 1

    def test_split_stores_value(self):
        d = self._single_note_case().to_dict()
        assert d["split"] == "test"

    def test_event_kind_stores_value(self):
        d = self._single_note_case().to_dict()
        assert d["event_kind"] == "single_note"

    def test_notes_serialize_correctly(self):
        d = self._single_note_case().to_dict()
        assert len(d["notes"]) == 1
        assert d["notes"][0]["midi"] == 40
        assert d["notes"][0]["string"] == 1

    def test_none_fields_excluded(self):
        d = self._single_note_case().to_dict()
        assert d.get("technique") is None
        assert d.get("technique_present") is None
        assert d.get("expected_onset_s") is None
        assert d.get("negative_reason") is None
        assert d.get("window_after_ms") is None

    def test_non_none_fields_present(self):
        case = self._single_note_case(
            technique="bend",
            technique_present=True,
            expected_onset_s=0.2,
            negative_reason="test negation",
            window_after_ms=400.0,
        )
        d = case.to_dict()
        assert d["technique"] == "bend"
        assert d["technique_present"] is True
        assert d["expected_onset_s"] == 0.2
        assert d["negative_reason"] == "test negation"
        assert d["window_after_ms"] == 400.0

    def test_metadata_and_context_serialized(self):
        case = self._single_note_case(
            metadata={"tuning": "E2", "pickup": "bridge"},
            technique_context={"type": "full"},
        )
        d = case.to_dict()
        assert d["metadata"] == {"tuning": "E2", "pickup": "bridge"}
        assert d["technique_context"] == {"type": "full"}

    def test_silence_case_roundtrip(self):
        case = CorpusCase(
            case_id="sil-001",
            audio_path="/audio/silence.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SILENCE,
            start_s=0.0,
            end_s=3.0,
            expected_present=False,
            negative_reason="silence recording",
        )
        d = case.to_dict()
        assert d["event_kind"] == "silence"
        assert d["expected_present"] is False


# ---------------------------------------------------------------------------
# Round-trip: to_dict → from_dict
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @staticmethod
    def _sample_case(**overrides) -> CorpusCase:
        kwargs = dict(
            case_id="rt-001",
            audio_path="/audio/rt.wav",
            source="roundtrip",
            split=CorpusSplit.CALIBRATION,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=47, string=3, fret=2, role="root"),),
            annotation_confidence=0.95,
            window_before_ms=150.0,
            window_after_ms=250.0,
        )
        kwargs.update(overrides)
        return CorpusCase(**kwargs)

    def test_minimal_roundtrip(self):
        case = self._sample_case()
        d = case.to_dict()
        restored = CorpusCase.from_dict(d)
        assert restored == case

    def test_full_roundtrip(self):
        case = self._sample_case(
            technique="harmonic",
            technique_present=True,
            expected_onset_s=0.1,
            negative_reason=None,
            metadata={"tuning": "standard"},
            technique_context={"type": "natural"},
        )
        d = case.to_dict()
        restored = CorpusCase.from_dict(d)
        assert restored == case

    def test_silence_roundtrip(self):
        case = CorpusCase(
            case_id="rt-sil",
            audio_path="/audio/sil.wav",
            source="rt",
            split=CorpusSplit.DEVELOPMENT,
            event_kind=EventKind.SILENCE,
            start_s=0.0,
            end_s=5.0,
            expected_present=False,
            negative_reason="ambient noise only",
        )
        d = case.to_dict()
        restored = CorpusCase.from_dict(d)
        assert restored == case

    def test_chord_roundtrip(self):
        case = CorpusCase(
            case_id="rt-chord",
            audio_path="/audio/chord.wav",
            source="rt",
            split=CorpusSplit.TEST,
            event_kind=EventKind.CHORD,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(
                CorpusExpectedNote(midi=40, string=1, role="root"),
                CorpusExpectedNote(midi=44, string=2, role="third"),
            ),
            window_before_ms=100.0,
        )
        d = case.to_dict()
        restored = CorpusCase.from_dict(d)
        assert restored == case

    def test_negative_case_roundtrip(self):
        case = CorpusCase(
            case_id="rt-neg",
            audio_path="/audio/neg.wav",
            source="rt",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=False,
            notes=(CorpusExpectedNote(midi=40),),
            negative_reason="note not played",
        )
        d = case.to_dict()
        restored = CorpusCase.from_dict(d)
        assert restored == case


# ---------------------------------------------------------------------------
# expected_midis
# ---------------------------------------------------------------------------


class TestExpectedMidis:
    def test_single_note(self):
        case = CorpusCase(
            case_id="midi-1",
            audio_path="/audio/a.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        assert case.expected_midis == (40,)

    def test_chord_notes(self):
        case = CorpusCase(
            case_id="midi-2",
            audio_path="/audio/b.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.CHORD,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(
                CorpusExpectedNote(midi=40),
                CorpusExpectedNote(midi=44),
                CorpusExpectedNote(midi=47),
            ),
        )
        assert case.expected_midis == (40, 44, 47)

    def test_empty_notes(self):
        """Silence cases have no expected midis."""
        case = CorpusCase(
            case_id="midi-3",
            audio_path="/audio/sil.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SILENCE,
            start_s=0.0,
            end_s=1.0,
            expected_present=False,
            negative_reason="silence",
        )
        assert case.expected_midis == ()


# ---------------------------------------------------------------------------
# resolve_audio_path
# ---------------------------------------------------------------------------


class TestResolveAudioPath:
    def test_absolute_path_unchanged(self):
        case = CorpusCase(
            case_id="abs-path",
            audio_path="/absolute/path/to/audio.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        result = case.resolve_audio_path(Path("/some/manifest.yaml"))
        assert result == Path("/absolute/path/to/audio.wav")

    def test_relative_path_resolved(self):
        case = CorpusCase(
            case_id="rel-path",
            audio_path="audio/test.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        manifest_path = Path("/data/manifests/corpus.yaml")
        result = case.resolve_audio_path(manifest_path)
        assert result == Path("/data/manifests/audio/test.wav")

    def test_relative_path_with_expanduser(self):
        case = CorpusCase(
            case_id="rel-expand",
            audio_path="~/corpus/notes.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        result = case.resolve_audio_path(Path("/mnt/manifest.yaml"))
        assert str(result).startswith("/" if Path.home().anchor else "")
        assert result.name == "notes.wav"

    def test_relative_path_resolves_below_tmp_path(self, tmp_path: Path):
        manifest_file = tmp_path / "subdir" / "manifest.yaml"
        manifest_file.parent.mkdir(parents=True)
        audio_rel = "audio_files/test.wav"
        case = CorpusCase(
            case_id="tmp-rel",
            audio_path=audio_rel,
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        result = case.resolve_audio_path(manifest_file)
        assert result == (manifest_file.parent / audio_rel).resolve()


# ---------------------------------------------------------------------------
# __post_init__ — validation
# ---------------------------------------------------------------------------


class TestPostInitValidation:
    """CorpusCase.__post_init__() validates fields on construction."""

    @staticmethod
    def _valid_kwargs(**overrides) -> dict:
        kwargs = dict(
            case_id="valid",
            audio_path="/audio/v.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        kwargs.update(overrides)
        return kwargs

    def test_empty_case_id(self):
        with pytest.raises(ValueError, match="case_id must not be empty"):
            CorpusCase(**self._valid_kwargs(case_id="  "))

    def test_negative_start_time(self):
        with pytest.raises(ValueError, match="invalid time range"):
            CorpusCase(**self._valid_kwargs(start_s=-1.0))

    def test_end_before_start(self):
        with pytest.raises(ValueError, match="invalid time range"):
            CorpusCase(**self._valid_kwargs(start_s=2.0, end_s=1.0))

    def test_end_equal_to_start(self):
        with pytest.raises(ValueError, match="invalid time range"):
            CorpusCase(**self._valid_kwargs(start_s=1.0, end_s=1.0))

    def test_confidence_too_low(self):
        with pytest.raises(ValueError, match="annotation_confidence"):
            CorpusCase(**self._valid_kwargs(annotation_confidence=-0.1))

    def test_confidence_too_high(self):
        with pytest.raises(ValueError, match="annotation_confidence"):
            CorpusCase(**self._valid_kwargs(annotation_confidence=1.1))

    def test_confidence_at_lower_bound(self):
        case = CorpusCase(**self._valid_kwargs(annotation_confidence=0.0))
        assert case.annotation_confidence == 0.0

    def test_confidence_at_upper_bound(self):
        case = CorpusCase(**self._valid_kwargs(annotation_confidence=1.0))
        assert case.annotation_confidence == 1.0

    def test_silence_with_notes(self):
        with pytest.raises(ValueError, match="silence cases cannot declare notes"):
            CorpusCase(**self._valid_kwargs(
                event_kind=EventKind.SILENCE,
                notes=(CorpusExpectedNote(midi=40),),
                expected_present=False,
                negative_reason="silence",
            ))

    def test_silence_with_technique(self):
        with pytest.raises(ValueError, match="silence cases cannot declare notes"):
            CorpusCase(**self._valid_kwargs(
                event_kind=EventKind.SILENCE,
                notes=(),
                technique="vibrato",
                expected_present=False,
                negative_reason="silence",
            ))

    def test_silence_with_expected_present_true(self):
        with pytest.raises(
            ValueError, match="silence cases must use expected_present=false"
        ):
            CorpusCase(**self._valid_kwargs(
                event_kind=EventKind.SILENCE,
                notes=(),
                expected_present=True,
            ))

    def test_non_silence_requires_notes(self):
        with pytest.raises(ValueError, match="case requires expected notes"):
            CorpusCase(**self._valid_kwargs(notes=()))

    def test_single_note_exactly_one(self):
        with pytest.raises(ValueError, match="single_note cases require exactly one"):
            CorpusCase(**self._valid_kwargs(
                notes=(CorpusExpectedNote(midi=40), CorpusExpectedNote(midi=44)),
                event_kind=EventKind.SINGLE_NOTE,
            ))

    def test_technique_requires_technique_field(self):
        with pytest.raises(ValueError, match="technique cases require technique"):
            CorpusCase(**self._valid_kwargs(
                event_kind=EventKind.TECHNIQUE,
                technique=None,
                technique_present=True,
                notes=(CorpusExpectedNote(midi=40),),
            ))

    def test_technique_requires_technique_present(self):
        with pytest.raises(
            ValueError, match="technique cases require technique_present"
        ):
            CorpusCase(**self._valid_kwargs(
                event_kind=EventKind.TECHNIQUE,
                technique="vibrato",
                technique_present=None,
                notes=(CorpusExpectedNote(midi=40),),
            ))

    def test_technique_present_without_technique(self):
        with pytest.raises(ValueError, match="technique_present requires technique"):
            CorpusCase(**self._valid_kwargs(
                technique_present=True,
            ))

    def test_negative_case_requires_reason(self):
        with pytest.raises(ValueError, match="negative cases require negative_reason"):
            CorpusCase(**self._valid_kwargs(
                expected_present=False,
                negative_reason=None,
            ))

    def test_negative_case_with_reason_ok(self):
        case = CorpusCase(**self._valid_kwargs(
            expected_present=False,
            negative_reason="wrong note",
        ))
        assert case.expected_present is False
        assert case.negative_reason == "wrong note"


# ---------------------------------------------------------------------------
# iter_manifest
# ---------------------------------------------------------------------------


class TestIterManifest:
    def test_yields_cases_from_lines(self, tmp_path: Path):
        manifest = tmp_path / "corpus.jsonl"
        lines = [
            json.dumps({
                "case_id": "c1",
                "audio_path": "/a/1.wav",
                "source": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 40}],
            }),
            json.dumps({
                "case_id": "c2",
                "audio_path": "/a/2.wav",
                "source": "test",
                "event_kind": "chord",
                "end_s": 2.0,
                "notes": [{"midi": 44}, {"midi": 47}],
            }),
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cases = list(iter_manifest(manifest))
        assert len(cases) == 2
        assert cases[0].case_id == "c1"
        assert cases[1].case_id == "c2"

    def test_skips_comments_and_blank_lines(self, tmp_path: Path):
        manifest = tmp_path / "commented.jsonl"
        lines = [
            "# Corpus manifest v1",
            "",
            "   ",
            json.dumps({
                "case_id": "c1",
                "audio_path": "/a/1.wav",
                "source": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 40}],
            }),
            "# trailing comment",
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cases = list(iter_manifest(manifest))
        assert len(cases) == 1
        assert cases[0].case_id == "c1"

    def test_duplicate_case_id_raises(self, tmp_path: Path):
        manifest = tmp_path / "dupes.jsonl"
        lines = [
            json.dumps({
                "case_id": "dup",
                "audio_path": "/a/1.wav",
                "source": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 40}],
            }),
            json.dumps({
                "case_id": "dup",
                "audio_path": "/a/2.wav",
                "source": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 42}],
            }),
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate case_id.*dup"):
            list(iter_manifest(manifest))

    def test_invalid_json_raises(self, tmp_path: Path):
        manifest = tmp_path / "invalid.jsonl"
        manifest.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid corpus case"):
            list(iter_manifest(manifest))

    def test_invalid_case_data_raises(self, tmp_path: Path):
        manifest = tmp_path / "badcase.jsonl"
        manifest.write_text(
            json.dumps({"case_id": "", "audio_path": "/x.wav", "event_kind": "single_note", "end_s": 1.0, "notes": [{"midi": 40}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid corpus case"):
            list(iter_manifest(manifest))

    def test_empty_file_yields_nothing(self, tmp_path: Path):
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("", encoding="utf-8")
        cases = list(iter_manifest(manifest))
        assert cases == []


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_loads_all_cases(self, tmp_path: Path):
        manifest = tmp_path / "all.jsonl"
        lines = [
            json.dumps({
                "case_id": "a",
                "audio_path": "/a.wav",
                "source": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 40}],
            }),
            json.dumps({
                "case_id": "b",
                "audio_path": "/b.wav",
                "source": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 42}],
            }),
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cases = load_manifest(manifest)
        assert len(cases) == 2

    def test_filter_by_split(self, tmp_path: Path):
        manifest = tmp_path / "filtered.jsonl"
        lines = [
            json.dumps({
                "case_id": "a",
                "audio_path": "/a.wav",
                "source": "test",
                "split": "calibration",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 40}],
            }),
            json.dumps({
                "case_id": "b",
                "audio_path": "/b.wav",
                "source": "test",
                "split": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 42}],
            }),
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        test_cases = load_manifest(manifest, split=CorpusSplit.TEST)
        assert len(test_cases) == 1
        assert test_cases[0].case_id == "b"

    def test_filter_by_development_split(self, tmp_path: Path):
        manifest = tmp_path / "dev.jsonl"
        lines = [
            json.dumps({
                "case_id": "x",
                "audio_path": "/x.wav",
                "source": "test",
                "split": "development",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 40}],
            }),
            json.dumps({
                "case_id": "y",
                "audio_path": "/y.wav",
                "source": "test",
                "split": "test",
                "event_kind": "single_note",
                "end_s": 1.0,
                "notes": [{"midi": 42}],
            }),
        ]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dev_cases = load_manifest(manifest, split=CorpusSplit.DEVELOPMENT)
        assert len(dev_cases) == 1
        assert dev_cases[0].case_id == "x"


# ---------------------------------------------------------------------------
# write_manifest → load_manifest round-trip
# ---------------------------------------------------------------------------


class TestWriteLoadRoundTrip:
    def test_write_and_reload(self, tmp_path: Path):
        cases = [
            CorpusCase(
                case_id="wrt-1",
                audio_path="/audio/1.wav",
                source="test",
                split=CorpusSplit.TEST,
                event_kind=EventKind.SINGLE_NOTE,
                start_s=0.0,
                end_s=1.5,
                expected_present=True,
                notes=(CorpusExpectedNote(midi=40),),
            ),
            CorpusCase(
                case_id="wrt-2",
                audio_path="/audio/2.wav",
                source="manual",
                split=CorpusSplit.DEVELOPMENT,
                event_kind=EventKind.CHORD,
                start_s=0.5,
                end_s=2.0,
                expected_present=False,
                notes=(
                    CorpusExpectedNote(midi=44, string=2),
                    CorpusExpectedNote(midi=47, string=3),
                ),
                negative_reason="chord not fully fretted",
            ),
        ]
        path = tmp_path / "written_manifest.jsonl"
        write_manifest(path, cases)
        assert path.exists()
        reloaded = load_manifest(path)
        assert len(reloaded) == 2
        assert reloaded[0] == cases[0]
        assert reloaded[1] == cases[1]

    def test_write_empty(self, tmp_path: Path):
        path = tmp_path / "empty_manifest.jsonl"
        write_manifest(path, [])
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert content == ""

    def test_append_mode(self, tmp_path: Path):
        path = tmp_path / "append_manifest.jsonl"
        case1 = CorpusCase(
            case_id="app-1",
            audio_path="/a/1.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        case2 = CorpusCase(
            case_id="app-2",
            audio_path="/a/2.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=42),),
        )
        write_manifest(path, [case1])
        write_manifest(path, [case2], append=True)
        reloaded = load_manifest(path)
        assert len(reloaded) == 2
        assert reloaded[0] == case1
        assert reloaded[1] == case2

    def test_writes_json_lines_format(self, tmp_path: Path):
        case = CorpusCase(
            case_id="fmt-1",
            audio_path="/a.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        path = tmp_path / "fmt_check.jsonl"
        write_manifest(path, [case])
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["case_id"] == "fmt-1"

    def test_creates_parent_directories(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "nested_manifest.jsonl"
        case = CorpusCase(
            case_id="nested",
            audio_path="/a.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        write_manifest(nested, [case])
        assert nested.exists()
        reloaded = load_manifest(nested)
        assert len(reloaded) == 1
        assert reloaded[0] == case

    def test_sorted_keys_in_output(self, tmp_path: Path):
        """write_manifest sorts the JSON keys so the output is deterministic."""
        case = CorpusCase(
            case_id="sorted",
            audio_path="/a.wav",
            source="test",
            split=CorpusSplit.TEST,
            event_kind=EventKind.SINGLE_NOTE,
            start_s=0.0,
            end_s=1.0,
            expected_present=True,
            notes=(CorpusExpectedNote(midi=40),),
        )
        path = tmp_path / "sorted.jsonl"
        write_manifest(path, [case])
        line = path.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert parsed == json.loads(json.dumps(parsed, sort_keys=True))
