"""Run the production verifier against a versioned evaluation corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pickhero import __version__
from pickhero.audio.chord_semantics import score_chord
from pickhero.audio.evidence import ExpectedNote
from pickhero.audio.match_mode import MatchMode
from pickhero.audio.verification_policy import VerificationPolicy
from pickhero.audio.verifier_composite import CompositeVerifier
from pickhero.evaluation.audio import AudioRepository
from pickhero.evaluation.manifest import CorpusCase, EventKind
from pickhero.evaluation.metrics import failing_records, summarize_records
from pickhero.evaluation.records import EvaluationRecord


@dataclass(frozen=True)
class EvaluationConfig:
    mode: MatchMode = MatchMode.JUDGE
    sample_rate: int = 48000
    onset_tolerance_ms: float = 45.0
    silence_threshold_db: float = -55.0
    require_onset_for_event: bool = False


@dataclass
class EvaluationRun:
    records: list[EvaluationRecord]
    summary: dict[str, object]

    def write(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "records.jsonl").open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        with (destination / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(self.summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with (destination / "failures.jsonl").open("w", encoding="utf-8") as stream:
            for record in failing_records(self.records):
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        (destination / "report.md").write_text(
            render_markdown_report(self.summary, self.records), encoding="utf-8"
        )


class CorpusEvaluator:
    def __init__(
        self,
        config: EvaluationConfig,
        *,
        verifier: CompositeVerifier | None = None,
        audio: AudioRepository | None = None,
    ) -> None:
        self.config = config
        self.verifier = verifier or CompositeVerifier(sample_rate=config.sample_rate)
        self.audio = audio or AudioRepository()
        self.policy = VerificationPolicy.from_mode(config.mode)

    def evaluate(
        self,
        cases: Iterable[CorpusCase],
        manifest_path: str | Path,
    ) -> EvaluationRun:
        records = [self.evaluate_case(case, manifest_path) for case in cases]
        summary = summarize_records(records)
        manifest = Path(manifest_path)
        summary["run"] = {
            "pickhero_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest),
            "manifest_sha256": (
                hashlib.sha256(manifest.read_bytes()).hexdigest()
                if manifest.exists()
                else None
            ),
            "config": {
                **asdict(self.config),
                "mode": self.config.mode.value,
            },
            "verification_policy": asdict(self.policy),
        }
        summary["mode"] = self.config.mode.value
        summary["sample_rate"] = self.config.sample_rate
        summary["onset_tolerance_ms"] = self.config.onset_tolerance_ms
        return EvaluationRun(records=records, summary=summary)

    def evaluate_case(
        self,
        case: CorpusCase,
        manifest_path: str | Path,
    ) -> EvaluationRecord:
        window = self.audio.window_for_case(
            case, manifest_path, self.config.sample_rate
        )
        health = window.health
        failures: list[str] = []
        if health.is_clipped:
            failures.append("audio_clipped")
        if health.has_dc_offset:
            failures.append("dc_offset")

        common = dict(
            case_id=case.case_id,
            source=case.source,
            split=case.split.value,
            event_kind=case.event_kind.value,
            mode=self.config.mode.value,
            audio_path=str(case.resolve_audio_path(Path(manifest_path))),
            start_s=case.start_s,
            end_s=case.end_s,
            expected_present=case.expected_present,
            expected_midis=case.expected_midis,
            annotation_confidence=case.annotation_confidence,
            metadata=dict(case.metadata),
            peak_dbfs=health.peak_dbfs,
            rms_dbfs=health.rms_dbfs,
            dc_offset=health.dc_offset,
            clipped_fraction=health.clipped_fraction,
        )

        if case.event_kind == EventKind.SILENCE:
            silent = self.verifier.verify_silence(
                window.samples, self.config.silence_threshold_db
            )
            if not silent:
                failures.append("unexpected_signal")
            return EvaluationRecord(
                **common,
                predicted_present=not silent,
                score=max(0.0, min(1.0, (health.rms_dbfs - self.config.silence_threshold_db) / 30.0)),
                failure_reasons=tuple(failures),
            )

        expected_notes = [
            ExpectedNote(
                midi=note.midi,
                string=note.string,
                fret=note.fret,
                event_id=case.case_id,
            )
            for note in case.notes
        ]

        if case.event_kind == EventKind.CHORD or len(expected_notes) > 1:
            verification = self.verifier.verify_chord(
                window.samples,
                expected_notes,
                self.config.mode,
                expected_onset_offset_ms=window.expected_onset_offset_ms,
                onset_tolerance_ms=self.config.onset_tolerance_ms,
            )
            chord = score_chord(
                expected_notes,
                verification,
                hit_threshold=self.policy.chord_hit_threshold,
                partial_threshold=self.policy.chord_partial_threshold,
                max_extra_for_hit=self.policy.max_extra_pitch_classes,
                max_strum_spread_ms=self.policy.max_strum_spread_ms,
            )
            onset_detected = any(note.is_onset_present for note in verification.notes)
            onset_ms = next(
                (note.onset_ms for note in verification.notes if note.onset_ms is not None),
                None,
            )
            predicted = chord.verdict == "hit"
            if self.config.require_onset_for_event:
                predicted = predicted and onset_detected
            if predicted != case.expected_present:
                failures.append("false_accept" if predicted else "false_reject")
            if chord.missing_critical_roles:
                failures.append("missing_critical_chord_tone")
            if chord.extra_pitch_classes:
                failures.append("foreign_pitch_class")
            return EvaluationRecord(
                **common,
                predicted_present=predicted,
                score=chord.score,
                onset_expected=window.expected_onset_offset_ms is not None,
                onset_detected=onset_detected,
                onset_error_ms=_onset_error(onset_ms, window.expected_onset_offset_ms),
                chord_verdict=chord.verdict,
                chord_score=chord.score,
                missing_roles=chord.missing_critical_roles,
                extra_pitch_classes=chord.extra_pitch_classes,
                failure_reasons=tuple(dict.fromkeys(failures)),
                details={"role_quality": chord.role_quality},
            )

        note = case.notes[0]
        verification = self.verifier.verify_single_note(
            window.samples,
            note.midi,
            self.config.mode,
            expected_onset_offset_ms=window.expected_onset_offset_ms,
            onset_tolerance_ms=self.config.onset_tolerance_ms,
        )
        predicted = verification.is_pitch_present
        if self.config.require_onset_for_event:
            predicted = predicted and verification.is_onset_present
        if predicted != case.expected_present:
            failures.append("false_accept" if predicted else "false_reject")
        if verification.alias_risk >= 0.8:
            failures.append("high_alias_risk")

        technique_detected: bool | None = None
        technique_uncertain: bool | None = None
        technique_quality: float | None = None
        technique_details: dict[str, object] = {}
        if case.technique:
            context = dict(case.technique_context)
            context.setdefault("midi_note", note.midi)
            context.setdefault("duration_ms", (case.end_s - case.start_s) * 1000.0)
            technique_result = self.verifier.verify_technique(
                window.samples, case.technique, context
            )
            technique_detected = technique_result.is_present
            technique_uncertain = technique_result.uncertain
            technique_quality = technique_result.quality
            technique_details = dict(technique_result.details)
            if (
                case.technique_present is not None
                and not technique_result.uncertain
                and technique_result.is_present != case.technique_present
            ):
                failures.append(
                    "technique_false_accept"
                    if technique_result.is_present
                    else "technique_false_reject"
                )

        cents_error = (
            verification.pitch_evidence.cents_error
            if verification.pitch_evidence is not None
            else None
        )
        return EvaluationRecord(
            **common,
            predicted_present=predicted,
            score=float(verification.confidence),
            onset_expected=window.expected_onset_offset_ms is not None,
            onset_detected=verification.is_onset_present,
            onset_error_ms=_onset_error(
                verification.onset_ms, window.expected_onset_offset_ms
            ),
            cents_error=cents_error,
            alias_risk=verification.alias_risk,
            technique=case.technique,
            technique_expected=case.technique_present,
            technique_detected=technique_detected,
            technique_uncertain=technique_uncertain,
            technique_quality=technique_quality,
            failure_reasons=tuple(dict.fromkeys(failures)),
            details={"technique": technique_details},
        )


def _onset_error(
    observed_ms: float | None,
    expected_ms: float | None,
) -> float | None:
    if observed_ms is None or expected_ms is None:
        return None
    return observed_ms - expected_ms


def render_markdown_report(
    summary: dict[str, object],
    records: list[EvaluationRecord],
) -> str:
    overall = summary["overall"]
    assert isinstance(overall, dict)
    event = overall["event"]
    onset = overall["absolute_onset_error_ms"]
    cents = overall["absolute_cents_error"]
    assert isinstance(event, dict) and isinstance(onset, dict) and isinstance(cents, dict)
    failures = failing_records(records)
    lines = [
        "# PickHero corpus evaluation",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Cases: **{summary['case_count']}**",
        f"- Event precision: **{float(event['precision']):.3f}**",
        f"- Event recall: **{float(event['recall']):.3f}**",
        f"- Event F1: **{float(event['f1']):.3f}**",
        f"- False-accept rate: **{float(event['false_accept_rate']):.3%}**",
        f"- Onset median absolute error: **{float(onset['median']):.1f} ms**",
        f"- Onset p95 absolute error: **{float(onset['p95']):.1f} ms**",
        f"- Median absolute pitch error: **{float(cents['median']):.1f} cents**",
        f"- Failures requiring inspection: **{len(failures)}**",
        "",
        "## Worst failures",
        "",
        "| Case | Kind | Expected | Predicted | Score | Audio | Reasons |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    ranked = sorted(
        failures,
        key=lambda record: (
            record.expected_present == record.predicted_present,
            -abs(record.score - 0.5),
        ),
    )[:50]
    for record in ranked:
        reasons = ", ".join(record.failure_reasons) or "metric failure"
        lines.append(
            f"| `{record.case_id}` | {record.event_kind} | "
            f"{record.expected_present} | {record.predicted_present} | "
            f"{record.score:.3f} | `{record.audio_path}` | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)
