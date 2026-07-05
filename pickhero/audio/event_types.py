"""Immutable event-snapshot types for cross-thread communication safety.

PerformanceEvent is mutable and shared by reference across threads (audio
callback → worker → stabilizer → matcher).  Its ``event_kind`` field can
change between detection, stabilization, queueing, and matching — causing
race conditions where the matcher sees the wrong event_kind.

These frozen dataclasses capture a snapshot of that state at emission time
so downstream consumers never see mid-flight mutations.

Types:

* :class:`RawAudioChunk` — raw PCM bytes + metadata from the audio callback.
* :class:`EventKindSnapshot` — immutable capture of a
  :class:`~pickhero.audio.performance.PerformanceEvent`'s taxonomy fields.
* :class:`PitchCandidate` — single-frame pitch observation with detector
  provenance.
* :class:`StableNoteEvent` — multi-frame consensus result that may carry
  an :class:`EventKindSnapshot` instead of (or alongside) the mutable
  :class:`~pickhero.audio.performance.PerformanceEvent`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickhero.audio.performance import TechniqueCandidate


@dataclass(frozen=True)
class RawAudioChunk:
    """A frozen slice of raw PCM audio from the audio callback.

    Parameters
    ----------
    samples : bytes
        Mono PCM bytes for this hop.
    sample_index : int
        Absolute sample offset within the stream.
    timestamp_ms : float
        Wall-clock timestamp in milliseconds.
    sample_rate : int
        Samples per second.
    """

    samples: bytes
    sample_index: int
    timestamp_ms: float
    sample_rate: int


@dataclass(frozen=True)
class EventKindSnapshot:
    """Immutable capture of a ``PerformanceEvent``'s taxonomy fields.

    Created at emission time.  Once instantiated the snapshot never changes,
    even if the original ``PerformanceEvent`` continues to mutate on the
    worker thread.  This prevents race conditions where the matcher sees a
    different ``event_kind`` than the detector emitted.

    Parameters
    ----------
    event_kind : str
        Taxonomy of the performed event (e.g. ``pick_onset``,
        ``legato_transition``, ``slide_landing``).
    technique_candidates : tuple[TechniqueCandidate, ...]
        Frozen copy of the candidate list at emission time.
    onset_ms : float
        Onset timestamp in milliseconds.
    midi_note : int | None
        Detached MIDI note number, or ``None`` if not yet resolved.
    confidence : float
        Detector confidence at emission time.
    """

    event_kind: str
    technique_candidates: tuple[TechniqueCandidate, ...]
    onset_ms: float
    midi_note: int | None
    confidence: float


@dataclass(frozen=True)
class PitchCandidate:
    """A single-frame pitch observation with detector provenance.

    Parameters
    ----------
    frequency : float
        Estimated frequency in Hz.
    confidence : float
        Detector confidence (0.0–1.0).
    midi_note : int
        Quantised MIDI note number.
    source : str
        Detector origin, e.g. ``"aubio_yin"``, ``"hybrid_f0"``,
        ``"chord_detector"``.
    onset : bool
        Whether this frame is the note onset.
    onset_sample : int | None
        Sample index of the onset, or ``None``.
    """

    frequency: float
    confidence: float
    midi_note: int
    source: str
    onset: bool
    onset_sample: int | None


@dataclass(frozen=True)
class StableNoteEvent:
    """A frozen multi-frame consensus result.

    Replaces the mutable ``StableNoteEvent`` in
    ``pickhero.audio.track_stabilizer`` when the event needs to travel
    across threads without risk of mid-flight mutation.

    Parameters
    ----------
    midi_note : int
        Stabilised MIDI note number.
    frequency : float
        Median stable frequency in Hz.
    confidence : float
        Mean confidence of consensus frames.
    name : str
        Scientific pitch notation (e.g. ``"E2"``).
    is_onset : bool
        Whether this is a picked onset or a continuation.
    onset_sample : int | None
        Sample index of onset, or ``None``.
    timestamp_ms : float
        Onset timestamp in milliseconds.
    consensus_frames : int
        Number of consecutive frames that agreed.
    event_snapshot : EventKindSnapshot | None
        Immutable taxonomy capture from the source
        ``PerformanceEvent``.  ``None`` when the event predates the
        snapshot mechanism.
    """

    midi_note: int
    frequency: float
    confidence: float
    name: str
    is_onset: bool
    onset_sample: int | None
    timestamp_ms: float
    consensus_frames: int
    event_snapshot: EventKindSnapshot | None = None
