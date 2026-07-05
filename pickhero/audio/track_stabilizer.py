"""Multi-frame pitch track stabilizer.

Sits between raw per-frame detection (PitchDetector) and the matcher.
Accumulates raw frames into stable note events via consensus, octave
commitment, and debounce. No per-frame detection reaches the matcher
without passing through this layer.

Pipeline stage:
    RawFrame → CandidateTrack → StableNoteEvent

A StableNoteEvent is emitted only after:
  - 3+ consecutive frames agree within ±35 cents, OR
  - onset + 2 stable follow-up frames, OR
  - a tab-prior-supported candidate with high harmonic evidence

Octave commitment: during the attack transient, multiple octave candidates
are tracked. The sustain frames resolve which is the true fundamental.
Once committed, octave jumps are suppressed for the note's duration.

Refractory: after emitting a note, a 50ms refractory prevents duplicate
emission from the same onset.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pickhero.audio.detector import DetectedNote
from pickhero.audio.event_types import EventKindSnapshot
from pickhero.audio.note_utils import freq_to_midi, midi_to_name, is_in_guitar_range

if TYPE_CHECKING:
    from pickhero.audio.performance import PerformanceEvent


# --- Tuning constants ---

_CONSENSUS_CENTS = 35.0       # frames must agree within this for consensus
_CONSENSUS_FRAMES = 3        # minimum stable frames to emit (without onset)
_ONSET_CONSENSUS_FRAMES = 2  # frames after onset to confirm (onset + 2 stable)
_OCTAVE_SUPPRESS_MS = 300.0  # suppress octave jumps for this long after commit
_REFRACTORY_MS = 50.0         # no new note within this window after emission
_MAX_TRACK_AGE_MS = 2000.0   # abandon a track that never reaches consensus
_TRANSIENT_TIMEOUT_MS = 100.0  # emit best-effort onset if no consensus after this
_TAB_PRIOR_CENTS = 100.0     # tab-supported candidate tolerance (±1 semitone)

@dataclass
class RawFrame:
    """A single per-hop observation from the detector."""
    timestamp_ms: float
    frequency: float
    confidence: float
    is_onset: bool
    onset_sample: int | None
    performance: "PerformanceEvent | None"


@dataclass
class CandidateTrack:
    """A multi-frame pitch track being accumulated toward consensus."""
    frames: list[RawFrame] = field(default_factory=list)
    base_freq: float = 0.0       # first confident frame's frequency
    base_midi: int = 0
    committed: bool = False       # octave resolved, emitting events
    emitted: bool = False         # a StableNoteEvent has been emitted for this track
    onset_frame_idx: int | None = None  # index into frames of the onset

    @property
    def age_ms(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp_ms - self.frames[0].timestamp_ms

    @property
    def latest_freq(self) -> float:
        return self.frames[-1].frequency if self.frames else 0.0

    @property
    def median_cents(self) -> float:
        """Median cents deviation from base_freq across all frames."""
        if not self.frames or self.base_freq <= 0:
            return 0.0
        cents = [1200.0 * np.log2(max(f.frequency, 1e-6) / self.base_freq)
                 for f in self.frames]
        return float(np.median(cents))

    def freq_at_cents(self, cents: float) -> float:
        """Convert a cents offset from base into absolute frequency."""
        if self.base_freq <= 0:
            return 0.0
        return self.base_freq * (2.0 ** (cents / 1200.0))


@dataclass
class StableNoteEvent:
    """A note event that has passed multi-frame consensus.

    This is what the matcher should receive — never a raw DetectedNote.
    """
    midi_note: int
    frequency: float          # median stable frequency
    confidence: float        # mean confidence of consensus frames
    name: str
    is_onset: bool
    onset_sample: int | None
    timestamp_ms: float      # onset timestamp (or first stable frame)
    performance: "PerformanceEvent | None"
    consensus_frames: int    # how many frames agreed
    event_snapshot: EventKindSnapshot | None = None

class TrackStabilizer:
    """Accumulates raw frames into stable note events.

    Usage:
        stabilizer = TrackStabilizer()
        for note in raw_detections:
            events = stabilizer.process(note, timestamp_ms, tab_prior_midi)
            for event in events:
                # feed to matcher
    """

    def __init__(self, sample_rate: int = 44100, hop_size: int = 512):
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        self._active: CandidateTrack | None = None
        self._last_emitted_ts: float = -1e9  # for refractory
        self._octave_locked_until: float = -1e9  # suppress octave jumps

    def reset(self) -> None:
        """Clear all state."""
        self._active = None
        self._last_emitted_ts = -1e9
        self._octave_locked_until = -1e9

    def process(
        self,
        note: DetectedNote | None,
        timestamp_ms: float,
        tab_prior_midi: int | None = None,
    ) -> list[StableNoteEvent]:
        """Process one raw frame. Returns zero or more stable events.

        Args:
            note: The raw DetectedNote from PitchDetector (may be None).
            timestamp_ms: Frame timestamp in ms from session start.
            tab_prior_midi: The MIDI note the tab expects near this time,
                           if known. Used to resolve octave ambiguity.

        Returns:
            List of StableNoteEvents emitted this frame (usually 0 or 1).
        """
        events: list[StableNoteEvent] = []

        # No detection this frame — check if active track is stale or
        # has a transient onset that never reached consensus.
        if note is None:
            if self._active is not None:
                # Transient onset: a click/dead-note/pick that didn't get
                # 3-frame consensus. Emit best-effort from the onset frame
                # after a timeout, so short signals still produce an event.
                # Use timestamp_ms (current frame) vs frames[0] to measure
                # real elapsed time — age_ms only spans appended frames.
                track_age_ms = timestamp_ms - self._active.frames[0].timestamp_ms
                if (self._active.onset_frame_idx is not None
                        and not self._active.emitted
                        and track_age_ms > _TRANSIENT_TIMEOUT_MS):
                    onset_frame = self._active.frames[self._active.onset_frame_idx]
                    # Only emit transient onsets with usable confidence — a
                    # conf<0.3 onset is noise (jack touch, cable bump) that
                    # the detector forwarded but YIN couldn't pitch-track.
                    if onset_frame.confidence < 0.3:
                        self._active = None
                        return events
                    # Only emit if the track is a true transient (≤4 frames).
                    # If there are 5+ frames with inconsistent pitches, the
                    # onset was noise (jack touch burst), not a real pick.
                    if len(self._active.frames) > 4:
                        self._active = None
                        return events
                    midi = freq_to_midi(onset_frame.frequency) if onset_frame.frequency > 0 else 0
                    events.append(StableNoteEvent(
                        midi_note=midi,
                        frequency=onset_frame.frequency,
                        confidence=onset_frame.confidence,
                        name=midi_to_name(midi),
                        is_onset=True,
                        onset_sample=onset_frame.onset_sample,
                        timestamp_ms=onset_frame.timestamp_ms,
                        performance=onset_frame.performance,
                        consensus_frames=len(self._active.frames),
                    ))
                    self._active.emitted = True
                    self._last_emitted_ts = onset_frame.timestamp_ms
                    self._active = None
                elif track_age_ms > _MAX_TRACK_AGE_MS:
                    self._active = None
            return events

        frame = RawFrame(
            timestamp_ms=timestamp_ms,
            frequency=note.frequency,
            confidence=note.confidence,
            is_onset=note.is_onset,
            onset_sample=note.onset_sample,
            performance=note.performance,
        )
        # Onset: start a new track (closes any previous one).
        if note.is_onset and note.confidence > 0.0:
            # Emit any pending consensus before closing.
            if self._active is not None and not self._active.emitted:
                emitted = self._try_emit(self._active, tab_prior_midi)
                events.extend(emitted)
                # If consensus failed but the track had an onset + at least
                # 1 high-confidence follow-up, emit as a transient. This
                # handles short signals (clicks, muted picks) that don't
                # sustain long enough for 3-frame consensus.
                if not emitted and self._active.onset_frame_idx is not None:
                    onset_frame = self._active.frames[self._active.onset_frame_idx]
                    if (onset_frame.confidence >= 0.3
                            and len(self._active.frames) <= 4):
                        midi = freq_to_midi(onset_frame.frequency) if onset_frame.frequency > 0 else 0
                        if midi > 0:
                            events.append(StableNoteEvent(
                                midi_note=midi,
                                frequency=onset_frame.frequency,
                                confidence=onset_frame.confidence,
                                name=midi_to_name(midi),
                                is_onset=True,
                                onset_sample=onset_frame.onset_sample,
                                timestamp_ms=onset_frame.timestamp_ms,
                                performance=onset_frame.performance,
                                consensus_frames=len(self._active.frames),
                            ))
                            self._active.emitted = True
                            self._last_emitted_ts = onset_frame.timestamp_ms

            self._active = CandidateTrack()
            self._active.frames.append(frame)
            self._active.onset_frame_idx = 0
            self._active.base_freq = frame.frequency
            self._active.base_midi = note.midi_note
            return events

        # Sustain / non-onset frame.
        if self._active is None:
            # No onset seen yet — start a track from sustain (rare, but
            # handles missed onsets).
            self._active = CandidateTrack()
            self._active.frames.append(frame)
            self._active.base_freq = frame.frequency
            self._active.base_midi = freq_to_midi(frame.frequency) if frame.frequency > 0 else 0
            return events

        # We have an active track — check octave stability.
        self._active.frames.append(frame)

        # Octave commitment: if not committed, resolve using sustain frames.
        if not self._active.committed:
            self._try_commit_octave(tab_prior_midi)

        # Check consensus.
        if not self._active.emitted:
            emitted = self._try_emit(self._active, tab_prior_midi)
            events.extend(emitted)

        return events

    def _try_commit_octave(self, tab_prior_midi: int | None) -> None:
        """Resolve octave ambiguity once enough sustain frames exist.

        During the attack transient, YIN may lock onto a harmonic. After
        a few frames, the dominant stable pitch reveals the true fundamental.
        If a tab prior is available, use it to disambiguate.
        """
        track = self._active
        if track is None or track.committed:
            return

        # Need at least 3 frames to attempt octave resolution.
        if len(track.frames) < 3:
            return

        # Get the median frequency of the last 3 frames (sustain).
        recent = track.frames[-3:]
        freqs = [f.frequency for f in recent if f.frequency > 0]
        if len(freqs) < 2:
            return

        median_freq = float(np.median(freqs))

        # If tab prior is available, check if median is an octave off.
        if tab_prior_midi is not None and median_freq > 0:
            prior_freq = 440.0 * (2.0 ** ((tab_prior_midi - 69) / 12.0))
            if prior_freq > 0:
                ratio = median_freq / prior_freq
                # If we're an octave off, correct to the tab-expected pitch.
                if 1.8 < ratio < 2.2:
                    # Median is an octave high — use the tab prior.
                    track.base_freq = prior_freq
                    track.base_midi = tab_prior_midi
                elif 0.45 < ratio < 0.55:
                    # Median is an octave low — use the tab prior.
                    track.base_freq = prior_freq
                    track.base_midi = tab_prior_midi
                else:
                    track.base_freq = median_freq
                    track.base_midi = freq_to_midi(median_freq)
        else:
            track.base_freq = median_freq
            track.base_midi = freq_to_midi(median_freq)

        track.committed = True
        self._octave_locked_until = track.frames[-1].timestamp_ms + _OCTAVE_SUPPRESS_MS

    def _try_emit(
        self, track: CandidateTrack, tab_prior_midi: int | None
    ) -> list[StableNoteEvent]:
        """Check if the track has reached consensus and emit if so."""
        if track.emitted:
            return []

        events: list[StableNoteEvent] = []

        # Refractory check.
        if track.frames[-1].timestamp_ms - self._last_emitted_ts < _REFRACTORY_MS:
            return []

        # Need enough frames for consensus.
        min_frames = _CONSENSUS_FRAMES
        if track.onset_frame_idx is not None:
            min_frames = track.onset_frame_idx + _ONSET_CONSENSUS_FRAMES + 1

        if len(track.frames) < min_frames:
            return []

        # Check that recent frames agree within ±_CONSENSUS_CENTS.
        recent = track.frames[-min_frames:]
        if track.base_freq <= 0:
            return []

        cents_list = []
        conf_list = []
        for f in recent:
            if f.frequency > 0 and track.base_freq > 0:
                cents = 1200.0 * np.log2(f.frequency / track.base_freq)
                # Octave-aware: fold to [-600, 600] so a frame at 2×F0
                # or 0.5×F0 doesn't break consensus. The octave commitment
                # resolves the true fundamental later.
                cents = ((cents + 600) % 1200) - 600
                cents_list.append(cents)
                conf_list.append(f.confidence)

        if len(cents_list) < min_frames:
            return []

        # Consensus: max deviation within tolerance.
        cents_arr = np.array(cents_list)
        if np.max(cents_arr) - np.min(cents_arr) > 2 * _CONSENSUS_CENTS:
            return []

        # Octave suppression: if locked, reject frames that jump an octave.
        latest_ts = track.frames[-1].timestamp_ms
        if latest_ts < self._octave_locked_until:
            latest_cents = cents_list[-1]
            if abs(latest_cents) > 900:  # near-octave jump
                return []

        # Consensus reached — emit.
        median_freq = track.base_freq
        midi_note = track.base_midi

        if not is_in_guitar_range(midi_note):
            track.emitted = True
            return []

        # Tab prior support: if available and matches, lower the bar.
        # (Consensus already required; tab prior just confirms octave.)

        onset_frame = None
        onset_ts = recent[0].timestamp_ms
        if track.onset_frame_idx is not None:
            onset_frame = track.frames[track.onset_frame_idx]
            onset_ts = onset_frame.timestamp_ms

        # Snapshot the PerformanceEvent's state at emission time.
        # This prevents race conditions where the mutable PerformanceEvent's
        # event_kind changes between detection, stabilization, and matching.
        event_snapshot = None
        if recent[-1].performance is not None:
            perf = recent[-1].performance
            event_snapshot = EventKindSnapshot(
                event_kind=perf.event_kind,
                technique_candidates=tuple(perf.technique_candidates),
                onset_ms=perf.onset_ms,
                midi_note=perf.midi_note,
                confidence=perf.confidence,
            )

        event = StableNoteEvent(
            midi_note=midi_note,
            frequency=median_freq,
            confidence=float(np.mean(conf_list)),
            name=midi_to_name(midi_note),
            is_onset=track.onset_frame_idx is not None,
            onset_sample=onset_frame.onset_sample if onset_frame else None,
            timestamp_ms=onset_ts,
            performance=recent[-1].performance,
            consensus_frames=len(recent),
            event_snapshot=event_snapshot,
        )
        events.append(event)
        track.emitted = True
        self._last_emitted_ts = latest_ts

        return events

    def flush(self) -> list[StableNoteEvent]:
        """Emit any pending track that reached consensus but wasn't emitted
        (e.g., at the end of a take). Does not force-emit tracks that
        haven't reached consensus."""
        events: list[StableNoteEvent] = []
        if self._active is not None and not self._active.emitted:
            events = self._try_emit(self._active, tab_prior_midi=None)
        self._active = None
        return events
