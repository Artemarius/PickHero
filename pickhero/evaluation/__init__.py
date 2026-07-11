"""Reproducible real-world evaluation support for PickHero."""

from pickhero.evaluation.calibration import calibrate_records, optimize_threshold
from pickhero.evaluation.manifest import (
    CorpusCase,
    CorpusExpectedNote,
    CorpusSplit,
    EventKind,
    load_manifest,
    write_manifest,
)
from pickhero.evaluation.metrics import summarize_records
from pickhero.evaluation.runner import CorpusEvaluator, EvaluationConfig, EvaluationRun

__all__ = [
    "CorpusCase",
    "CorpusExpectedNote",
    "CorpusEvaluator",
    "CorpusSplit",
    "EvaluationConfig",
    "EvaluationRun",
    "EventKind",
    "calibrate_records",
    "load_manifest",
    "optimize_threshold",
    "summarize_records",
    "write_manifest",
]
