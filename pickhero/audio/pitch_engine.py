"""High-accuracy pitch detection engine with worker-thread consensus.

Runs in a worker thread fed from the audio callback's ring buffer. Produces
richer PitchCandidate results with cents error, confidence, and source flags.

The audio callback submits hop-sized chunks with their absolute stream sample
index. The worker drains the queue and runs the consensus pipeline. Results
carry their own chunk_start_sample so timestamps are stable regardless of when
the callback drains the output queue.

Profiles:
- "portable": uses PitchDetector (aubio yinfast) — no worker needed.
- "high_accuracy": uses PitchEngine with multi-resolution YIN + spectral checks.
The audio callback must NEVER block on the worker. If the ring buffer is full,
the oldest buffer is dropped (xrun logged).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from pickhero.audio.detector import PitchDetector, DetectedNote
from pickhero.audio.log_frequency import MultiResolutionLogSpectrum
from pickhero.audio.note_utils import freq_to_midi, freq_to_cents_deviation, midi_to_name
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickhero.audio.performance import PerformanceEvent

@dataclass
class PitchCandidate:
    """A pitch estimate with rich metadata from the consensus pipeline."""
    best_midi: int | None
    cents_error: float | None
    raw_frequency: float
    confidence: float
    source_flags: set[str] = field(default_factory=set)  # detector/evidence sources
    harmonic_support: float = 0.0
    subharmonic_risk: float = 0.0

    def to_detected_note(self, is_onset: bool = False, onset_sample: int | None = None,
                         performance=None) -> DetectedNote | None:
        """Convert to a DetectedNote for the existing matcher pipeline. None if no pitch.

        ``performance`` is the PerformanceEvent captured on the worker thread at
        processing time (Patch 1a). The consensus midi/confidence were stamped
        onto it on the worker thread (Patch 1a race fix) before enqueuing — this
        method must NOT mutate it, to avoid a cross-thread write race with the
        worker's f0_curve/energy_envelope appends.
        """
        if self.best_midi is None or self.raw_frequency <= 0:
            return None
        return DetectedNote(
            midi_note=self.best_midi,
            frequency=self.raw_frequency,
            confidence=self.confidence,
            name=midi_to_name(self.best_midi),
            is_onset=is_onset,
            onset_sample=onset_sample,
            performance=performance,
        )


@dataclass
class _WorkItem:
    """A chunk submitted by the audio callback, with its stream sample index."""
    chunk: np.ndarray
    chunk_start_sample: int


@dataclass
class _EngineResult:
    """A worker result carrying its own stable timestamp metadata."""
    candidate: PitchCandidate
    is_onset: bool
    onset_sample: int | None
    chunk_start_sample: int
    performance: "PerformanceEvent | None" = None


class PitchEngine:
    """Worker-thread pitch engine for the HighAccuracy profile.

    Fed from the audio callback via a ring buffer (queue.Queue with maxsize).
    A worker thread drains the queue and runs the consensus pipeline:
    1. YIN/yinfast pitch estimate (existing aubio.pitch) — onset called ONCE
    2. Multi-resolution: optional second YIN at a larger window for low notes
    3. Spectral check on a rolling 4096/8192 analysis buffer (not the hop chunk)
    4. Tab-guided candidate check as a prior only (never as proof)
    5. Cents error stored
    6. Confidence combines signal level, stability, harmonic support

    The audio callback must not block on the worker. If the queue is full,
    the oldest buffer is dropped (xrun logged via the xrun counter).
    """

    # Long enough to resolve low guitar/bass fundamentals. High-note analysis
    # still uses shorter windows inside MultiResolutionLogSpectrum.
    _SPECTRAL_BUF_SIZE = 16384

    def __init__(
        self,
        sample_rate: int = 48000,
        hop_size: int = 256,
        buf_size: int = 4096,
        confidence_threshold: float = 0.8,
        onset_threshold: float = 0.3,
        noise_gate_db: float = -60.0,
        calibration=None,
        profile: str = "high_accuracy",
        ml_model_path: str = "",
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.buf_size = buf_size
        self.confidence_threshold = confidence_threshold
        self.onset_threshold = onset_threshold
        self.noise_gate_db = noise_gate_db
        self.profile = profile

        # The primary PitchDetector does the actual aubio processing (onset + YIN).
        # It is the ONLY thing that touches aubio's onset detector — no double calls.
        self._detector = PitchDetector(
            buf_size=buf_size,
            hop_size=hop_size,
            sample_rate=sample_rate,
            confidence_threshold=confidence_threshold,
            onset_threshold=onset_threshold,
            noise_gate_db=noise_gate_db,
            calibration=calibration if calibration else None,
        )

        # Second YIN detector at a larger window for multi-resolution.
        # Low guitar notes (E2 = 82 Hz) need a long window for reliable
        # pitch tracking. Always create it — even when the primary buffer
        # is already 4096, a 8192 window gives 5.9 Hz/bin resolution for
        # distinguishing E2 (82Hz) from E1 (41Hz) and E3 (165Hz).
        # Runs in parallel (no added latency to primary).
        self._detector_large: PitchDetector | None = None
        if sample_rate >= 44100:
            large_buf = max(8192, buf_size * 2)
            try:
                self._detector_large = PitchDetector(
                    buf_size=large_buf,
                    hop_size=hop_size,
                    sample_rate=sample_rate,
                    confidence_threshold=confidence_threshold,
                    onset_threshold=onset_threshold,
                    noise_gate_db=noise_gate_db,
                    calibration=calibration if calibration else None,
                )
            except Exception:
                self._detector_large = None

        # Rolling spectral analysis buffer for the validator.
        self._spectral_buf = np.zeros(self._SPECTRAL_BUF_SIZE, dtype=np.float32)
        self._spectral_fill = 0
        self._spectral_window = np.hanning(self._SPECTRAL_BUF_SIZE).astype(np.float32)
        self._spectral_freqs = np.fft.rfftfreq(
            self._SPECTRAL_BUF_SIZE, 1.0 / sample_rate
        )
        self._log_front_end = MultiResolutionLogSpectrum(
            sample_rate=sample_rate,
            min_midi=24,
            max_midi=108,
            fft_sizes=(4096, 8192, self._SPECTRAL_BUF_SIZE),
        )
        self._last_committed_midi: int | None = None
        self._last_commit_confidence: float = 0.0

        # Ring buffer: audio callback pushes work items, worker drains.
        self._in_queue: queue.Queue[_WorkItem] = queue.Queue(maxsize=64)
        # Output queue: worker pushes _EngineResult, main thread drains.
        self._out_queue: queue.Queue[_EngineResult] = queue.Queue(maxsize=256)
        # Tab prior: expected MIDI notes near current playback position.
        self._tab_prior: set[int] = set()
        self._tab_prior_lock = threading.Lock()

        self._xrun_count = 0
        self._running = False
        self._thread: threading.Thread | None = None

        # ML assist is not bundled.  The config path ``ml_model_path`` is
        # reserved for users who provide their own ONNX model.  When no
        # model is provided, ``_ml_assist`` stays None so callers can
        # check for ML availability without importing the optional dep.
        self._ml_model_path = ml_model_path
        self._ml_assist = None

    @property
    def detector(self) -> PitchDetector:
        """Expose the underlying PitchDetector for portable-mode compatibility."""
        return self._detector

    @property
    def last_signal_db(self) -> float:
        return self._detector.last_signal_db

    @property
    def last_freq(self) -> float:
        return self._detector.last_freq

    @property
    def last_confidence(self) -> float:
        return self._detector.last_confidence

    def set_noise_gate_db(self, db: float) -> None:
        self.noise_gate_db = db
        self._detector.set_noise_gate_db(db)
        if self._detector_large:
            self._detector_large.set_noise_gate_db(db)

    def get_onset_delay(self) -> int:
        return self._detector.get_onset_delay()

    def reset(self) -> None:
        self._detector.reset()
        if self._detector_large:
            self._detector_large.reset()
        self._spectral_buf[:] = 0
        self._spectral_fill = 0
        self._last_committed_midi = None
        self._last_commit_confidence = 0.0

    def set_tab_prior(self, midi_notes: set[int]) -> None:
        """Set expected MIDI notes near current playback position (thread-safe)."""
        with self._tab_prior_lock:
            self._tab_prior = set(midi_notes)

    def submit(self, chunk: np.ndarray, chunk_start_sample: int = 0) -> None:
        """Submit a hop-sized audio chunk from the audio callback.

        chunk_start_sample is the absolute stream sample index of this chunk.
        It is carried through to the output so timestamps are stable.

        Non-blocking: if the queue is full, drops the oldest buffer (xrun).
        Must be safe to call from the real-time audio thread.
        """
        item = _WorkItem(chunk=chunk.copy(), chunk_start_sample=chunk_start_sample)
        try:
            self._in_queue.put_nowait(item)
        except queue.Full:
            try:
                self._in_queue.get_nowait()
                self._in_queue.put_nowait(item)
                self._xrun_count += 1
            except queue.Empty:
                pass

    def get_xrun_count(self) -> int:
        return self._xrun_count

    def get_candidates(self) -> list[_EngineResult]:
        """Drain all pending results from the output queue (non-blocking).

        Each result carries its own chunk_start_sample for stable timestamps.
        """
        results = []
        while True:
            try:
                results.append(self._out_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def process_sync(self, chunk: np.ndarray, chunk_start_sample: int = 0) -> _EngineResult | None:
        """Process a chunk synchronously on the calling thread.

        Used by the unified worker when the engine and the worker share a
        single thread — avoids the async submit/get_candidates race where
        get_candidates returns empty because the engine's own worker hasn't
        processed the chunk yet.
        """
        result = self._process_chunk(chunk)
        if result is None:
            return None
        perf = None
        art = getattr(self._detector, "_articulation", None)
        if art is not None:
            perf = art.active_event
            cand = result.candidate
            if perf is not None and cand.best_midi is not None and cand.raw_frequency > 0:
                perf.midi_note = cand.best_midi
                perf.confidence = cand.confidence
        return _EngineResult(
            candidate=result.candidate,
            is_onset=result.is_onset,
            onset_sample=result.onset_sample,
            chunk_start_sample=chunk_start_sample,
            performance=perf,
        )

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _worker_loop(self) -> None:
        """Worker thread: drain input queue, run consensus, push to output queue.

        The onset detector is called EXACTLY ONCE per chunk, inside
        PitchDetector.process(). The worker does NOT call _onset() again.
        """
        while self._running:
            try:
                item = self._in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            result = self._process_chunk(item.chunk)
            if result is not None:
                # Capture the active PerformanceEvent on the worker thread —
                # the only place "active during processing" is true. Reading it
                # later, at callback-drain time, would attach the *current*
                # sounding note's event, not the one that was sounding when
                # this chunk was processed.
                perf = None
                art = getattr(self._detector, "_articulation", None)
                if art is not None:
                    perf = art.active_event
                    # Stamp the consensus midi/confidence on the worker thread
                    # (Patch 1a race fix): all writes to the PerformanceEvent
                    # happen here, on the worker. The queue provides the
                    # happens-before guarantee for the main thread's reads in
                    # to_detected_note, which no longer mutates it.
                    cand = result.candidate
                    if perf is not None and cand.best_midi is not None and cand.raw_frequency > 0:
                        perf.midi_note = cand.best_midi
                        perf.confidence = cand.confidence
                engine_result = _EngineResult(
                    candidate=result.candidate,
                    is_onset=result.is_onset,
                    onset_sample=result.onset_sample,
                    chunk_start_sample=item.chunk_start_sample,
                    performance=perf,
                )
                try:
                    self._out_queue.put_nowait(engine_result)
                except queue.Full:
                    try:
                        self._out_queue.get_nowait()
                        self._out_queue.put_nowait(engine_result)
                    except queue.Empty:
                        pass

    def _process_chunk(self, chunk: np.ndarray) -> _EngineResult | None:
        """Run multi-resolution pitch consensus on one hop-sized chunk.

        YIN remains the low-latency observation, while a harmonic-sieve front
        end contributes several fundamental hypotheses from the rolling audio
        window.  The resolver can therefore reject octave aliases instead of
        merely validating whichever single pitch YIN produced first.
        """
        self._push_spectral(chunk)

        candidates: list[PitchCandidate] = []
        is_onset = False
        onset_sample: int | None = None
        primary_freq = 0.0
        primary_midi: int | None = None
        primary_confidence = 0.0

        with self._tab_prior_lock:
            tab_prior = set(self._tab_prior)

        result = self._detector.process(chunk)
        if result is not None:
            primary_freq = float(result.frequency)
            primary_midi = int(result.midi_note) if result.midi_note > 0 else None
            primary_confidence = float(result.confidence)
            is_onset = bool(result.is_onset)
            onset_sample = result.onset_sample

        if primary_freq > 0.0 and primary_midi is not None:
            selected_freq = primary_freq
            selected_midi = primary_midi
            selected_confidence = primary_confidence
            flags: set[str] = {"yin", "yin_primary"}

            if self._detector_large is not None and primary_freq < 220.0:
                try:
                    large = self._detector_large.process(chunk)
                except Exception:
                    large = None
                if large is not None and large.frequency > 0 and large.midi_note > 0:
                    candidates.append(PitchCandidate(
                        best_midi=int(large.midi_note),
                        cents_error=freq_to_cents_deviation(float(large.frequency))[1],
                        raw_frequency=float(large.frequency),
                        confidence=float(large.confidence) * 0.96,
                        source_flags={"yin_large"},
                    ))
                    if large.confidence > selected_confidence + 0.04:
                        selected_freq = float(large.frequency)
                        selected_midi = int(large.midi_note)
                        selected_confidence = float(large.confidence)
                        flags.add("yin_large")
                    if not is_onset and large.is_onset:
                        is_onset = True
                        onset_sample = large.onset_sample

            _, cents = freq_to_cents_deviation(selected_freq)
            candidates.append(PitchCandidate(
                best_midi=selected_midi,
                cents_error=cents,
                raw_frequency=selected_freq,
                confidence=selected_confidence,
                source_flags=flags,
            ))

        # The multi-resolution front end sees a long rolling window and returns
        # several mutually competing fundamentals. It is useful even when YIN
        # reports no pitch during a noisy attack.
        if self._spectral_fill >= 4096:
            window = self._spectral_buf[:self._spectral_fill]
            frame = self._log_front_end.analyse(
                window,
                top_k=5,
                prior_midis=tab_prior,
            )
            for hypothesis in frame.hypotheses:
                confidence = (
                    0.12
                    + hypothesis.salience * 0.62
                    + hypothesis.harmonic_support * 0.26
                    - hypothesis.subharmonic_risk * 0.16
                )
                candidates.append(PitchCandidate(
                    best_midi=hypothesis.midi,
                    cents_error=hypothesis.cents_error,
                    raw_frequency=hypothesis.frequency,
                    confidence=max(0.0, min(1.0, confidence)),
                    source_flags={"harmonic_sieve", "multi_resolution"},
                    harmonic_support=hypothesis.harmonic_support,
                    subharmonic_risk=hypothesis.subharmonic_risk,
                ))

        if not candidates:
            return _EngineResult(
                candidate=PitchCandidate(
                    best_midi=None,
                    cents_error=None,
                    raw_frequency=primary_freq,
                    confidence=0.0,
                    source_flags=set(),
                ),
                is_onset=is_onset,
                onset_sample=onset_sample,
                chunk_start_sample=0,
            )

        if tab_prior:
            candidates = self._apply_tab_prior(candidates, tab_prior)

        reference_midi = primary_midi
        if reference_midi is None:
            pitched = [candidate for candidate in candidates if candidate.best_midi is not None]
            if pitched:
                reference_midi = max(pitched, key=lambda candidate: candidate.confidence).best_midi
        tab_prior_midi = None
        if tab_prior and reference_midi is not None:
            tab_prior_midi = min(tab_prior, key=lambda midi: abs(midi - reference_midi))

        best = self._resolve_candidates(candidates, tab_prior_midi)
        if best is None or best.best_midi is None or best.raw_frequency <= 0.0:
            return _EngineResult(
                candidate=PitchCandidate(
                    best_midi=None,
                    cents_error=None,
                    raw_frequency=primary_freq,
                    confidence=0.0,
                    source_flags=set(),
                ),
                is_onset=is_onset,
                onset_sample=onset_sample,
                chunk_start_sample=0,
            )

        flags_out = set(best.source_flags)
        spectral_available = self._spectral_fill >= self._SPECTRAL_BUF_SIZE // 2
        harmonic_score = best.harmonic_support or (
            self._spectral_check(best.raw_frequency) if spectral_available else 0.0
        )
        final_confidence = float(best.confidence)
        if spectral_available:
            if harmonic_score < 0.08:
                final_confidence *= 0.42
            elif harmonic_score < 0.16:
                final_confidence *= 0.68
            else:
                final_confidence = min(1.0, final_confidence * (0.88 + harmonic_score * 0.20))
            final_confidence *= 1.0 - min(0.28, best.subharmonic_risk * 0.22)

        if final_confidence >= 0.28:
            self._last_committed_midi = best.best_midi
            self._last_commit_confidence = final_confidence

        return _EngineResult(
            candidate=PitchCandidate(
                best_midi=best.best_midi,
                cents_error=best.cents_error,
                raw_frequency=best.raw_frequency,
                confidence=max(0.0, min(1.0, final_confidence)),
                source_flags=flags_out,
                harmonic_support=harmonic_score,
                subharmonic_risk=best.subharmonic_risk,
            ),
            is_onset=is_onset,
            onset_sample=onset_sample,
            chunk_start_sample=0,
        )

    def _find_spectral_peak(self, primary_freq: float) -> float | None:
        """Find the strongest FFT peak near the primary frequency.

        Returns None if the buffer is not full enough for a reliable check.
        """
        if self._spectral_fill < self._SPECTRAL_BUF_SIZE // 2:
            return None

        buf = self._spectral_buf[:self._SPECTRAL_BUF_SIZE] * self._spectral_window
        spectrum = np.abs(np.fft.rfft(buf))

        if len(spectrum) == 0:
            return None

        # Search in a ±3-semitone window around the primary
        lo_freq = primary_freq * (2 ** (-3 / 12))
        hi_freq = primary_freq * (2 ** (3 / 12))
        lo_idx = max(0, int(np.searchsorted(self._spectral_freqs, lo_freq)))
        hi_idx = min(len(spectrum) - 1, int(np.searchsorted(self._spectral_freqs, hi_freq)))

        if hi_idx <= lo_idx:
            return None

        window = spectrum[lo_idx:hi_idx + 1]
        peak_idx = lo_idx + int(np.argmax(window))
        peak_freq = float(self._spectral_freqs[peak_idx])

        # Use harmonic likelihood instead of the old 0.005 energy-ratio
        # threshold. A real guitar note has harmonics at 2×-5×F0; a cable
        # resonance or noise doesn't. Score < 0.05 → reject.
        score = self._spectral_check(peak_freq)
        if score < 0.05:
            return None
        return peak_freq

    def _apply_tab_prior(self, candidates: list[PitchCandidate], tab_prior: set[int]) -> list[PitchCandidate]:
        """Boost confidence of candidates matching the tab prior."""
        if not tab_prior:
            return candidates

        for cand in candidates:
            if cand.best_midi is not None and cand.best_midi in tab_prior:
                cand.source_flags = cand.source_flags | {"tab_prior"}
                cand.confidence = min(1.0, cand.confidence + 0.05)
        return candidates

    def _resolve_candidates(
        self,
        candidates: list[PitchCandidate],
        tab_prior_midi: int | None,
    ) -> PitchCandidate | None:
        """Resolve YIN and harmonic-sieve hypotheses without octave snap.

        Evidence is grouped by semitone. Agreement between independent sources
        is rewarded, while subharmonic risk and unsupported octave jumps are
        penalised. Tab context and temporal continuity remain priors only.
        """
        valid = [
            candidate for candidate in candidates
            if candidate.best_midi is not None
            and 24 <= candidate.best_midi <= 108
            and candidate.raw_frequency > 0.0
            and candidate.confidence >= 0.08
        ]
        if not valid:
            return None

        groups: dict[int, list[PitchCandidate]] = {}
        for candidate in valid:
            groups.setdefault(int(candidate.best_midi), []).append(candidate)

        def source_family(candidate: PitchCandidate) -> str:
            if "harmonic_sieve" in candidate.source_flags:
                return "harmonic"
            if "yin_large" in candidate.source_flags:
                return "yin_large"
            if "yin" in candidate.source_flags:
                return "yin"
            if "spectral" in candidate.source_flags:
                return "spectral"
            return "other"

        def group_score(midi: int, group: list[PitchCandidate]) -> float:
            best = max(group, key=lambda item: item.confidence)
            confidence = max(item.confidence for item in group)
            families = {source_family(item) for item in group}
            agreement = min(0.18, max(0, len(families) - 1) * 0.09)
            harmonic = max(item.harmonic_support for item in group)
            risk = max(item.subharmonic_risk for item in group)
            score = confidence + agreement + harmonic * 0.10 - risk * 0.18

            if tab_prior_midi is not None:
                distance = abs(midi - tab_prior_midi)
                if distance <= 1:
                    score += 0.12
                elif distance <= 3:
                    score += 0.05
                elif distance <= 12:
                    score += 0.03
                elif distance > 12:
                    score -= 0.05

            if self._last_committed_midi is not None:
                distance = abs(midi - self._last_committed_midi)
                if distance <= 1:
                    score += 0.11 * max(0.4, self._last_commit_confidence)
                elif distance == 12:
                    score -= 0.12 * max(0.4, self._last_commit_confidence)
                elif distance > 12:
                    score -= min(0.16, (distance - 12) * 0.01)

            # Prefer a lower fundamental when an upper candidate is explicitly
            # marked as its likely harmonic and their evidence is otherwise close.
            lower_group = groups.get(midi - 12)
            if lower_group:
                lower_conf = max(item.confidence for item in lower_group)
                if risk >= 0.45 and lower_conf >= confidence - 0.14:
                    score -= 0.14
            return score

        winning_midi, winning_group = max(
            groups.items(), key=lambda item: group_score(item[0], item[1])
        )

        # Prefer the most accurate frequency estimate from the winning group.
        # Harmonic-sieve candidates provide refined cents; YIN remains favoured
        # when confidence is materially higher.
        winner = max(
            winning_group,
            key=lambda item: (
                item.confidence
                + item.harmonic_support * 0.10
                - item.subharmonic_risk * 0.10
            ),
        )
        merged_flags: set[str] = set()
        for item in winning_group:
            merged_flags.update(item.source_flags)
        winner.source_flags = merged_flags
        winner.harmonic_support = max(item.harmonic_support for item in winning_group)
        winner.subharmonic_risk = max(item.subharmonic_risk for item in winning_group)
        winner.confidence = max(item.confidence for item in winning_group)
        return winner

    def _push_spectral(self, chunk: np.ndarray) -> None:
        """Push audio into the rolling spectral analysis buffer."""
        n = len(chunk)
        if n == 0:
            return
        if n >= self._SPECTRAL_BUF_SIZE:
            self._spectral_buf[:] = chunk[-self._SPECTRAL_BUF_SIZE:]
            self._spectral_fill = self._SPECTRAL_BUF_SIZE
        elif self._spectral_fill + n <= self._SPECTRAL_BUF_SIZE:
            self._spectral_buf[self._spectral_fill:self._spectral_fill + n] = chunk
            self._spectral_fill += n
        else:
            # Shift left and append
            keep = self._SPECTRAL_BUF_SIZE - n
            self._spectral_buf[:keep] = self._spectral_buf[self._spectral_fill - keep:self._spectral_fill]
            self._spectral_buf[keep:] = chunk
            self._spectral_fill = self._SPECTRAL_BUF_SIZE

    def _spectral_check(self, freq: float) -> float:
        """Harmonic likelihood score for a candidate fundamental frequency.

        Returns a float in [0, 1] indicating how well the spectrum supports
        freq as a fundamental. The previous check used a 0.005 energy-ratio
        threshold that was essentially a no-op (any signal with energy near
        the candidate passed).

        The model:
          1. Measure F0 peak energy (±5% around freq).
          2. Sum weighted energy at harmonics 2×-5× F0.
          3. Estimate noise floor as median of the spectrum.
          4. harmonic_ratio = (harmonic_energy - noise_floor * n_harmonics)
                              / (f0_energy + harmonic_energy)
          5. Apply spectral flatness penalty (broadband noise → low score).

        A real guitar note scores > 0.15. A cable resonance or pure tone
        with no harmonic series scores < 0.05.
        """
        if freq <= 0 or self._spectral_fill < self._SPECTRAL_BUF_SIZE // 2:
            return 0.0

        buf = self._spectral_buf[:self._SPECTRAL_BUF_SIZE] * self._spectral_window
        spectrum = np.abs(np.fft.rfft(buf))
        if len(spectrum) == 0 or float(np.sum(spectrum)) <= 0:
            return 0.0

        nyquist = self.sample_rate / 2.0

        def _peak_energy(center: float, tol: float = 0.05) -> float:
            # Narrow center band for the peak; wider sidebands (excluding the
            # center) provide a local median baseline. This rejects broadband
            # noise without erasing narrow harmonic peaks at low frequencies
            # where the ±5% band may contain only a few bins.
            c_lo = center * 0.99
            c_hi = center * 1.01
            s_lo = center * (1.0 - tol)
            s_hi = center * (1.0 + tol)
            center_mask = (self._spectral_freqs >= c_lo) & (self._spectral_freqs <= c_hi)
            side_mask = ((self._spectral_freqs >= s_lo) & (self._spectral_freqs < c_lo)) | (
                (self._spectral_freqs > c_hi) & (self._spectral_freqs <= s_hi)
            )
            if not np.any(center_mask):
                return 0.0
            peak = float(np.max(spectrum[center_mask]))
            if np.any(side_mask):
                baseline = float(np.median(spectrum[side_mask]))
            else:
                baseline = 0.0
            return max(0.0, peak - baseline)

        # F0 peak energy
        f0_energy = _peak_energy(freq)
        if f0_energy <= 0:
            return 0.0

        # Sum weighted energy at harmonics 2×-5× F0
        weights = [1.0, 0.7, 0.5, 0.3]
        harmonic_energy = 0.0
        n_harmonics_found = 0
        h = 2
        while freq * h < nyquist and h <= 5:
            e = _peak_energy(freq * h)
            harmonic_energy += e * weights[h - 2]
            if e > 0:
                n_harmonics_found += 1
            h += 1

        # Noise floor estimate: median of spectrum (robust to harmonic peaks)
        noise_floor = float(np.median(spectrum))

        # Harmonic ratio: harmonic energy above noise floor, normalized by
        # total peak energy (F0 + harmonics). This is robust to added noise
        # because noise inflates the median floor, which we subtract.
        total_peak = f0_energy + harmonic_energy
        noise_contribution = noise_floor * max(n_harmonics_found, 1) * 2.0
        harmonic_above_noise = max(0.0, harmonic_energy - noise_contribution)
        harmonic_ratio = harmonic_above_noise / total_peak if total_peak > 0 else 0.0

        # Spectral flatness penalty: broadband noise has high flatness
        eps = 1e-12
        geo = np.exp(np.mean(np.log(np.maximum(spectrum[1:], eps))))
        ari = np.mean(spectrum[1:])
        flatness = float(geo / ari) if ari > eps else 1.0

        # Final score: harmonic ratio reduced by noise penalty.
        # flatness ** 1.5 sharply penalizes broadband noise (flatness ~0.85)
        # while barely affecting harmonic signals (flatness ~0.15).
        score = harmonic_ratio * (1.0 - flatness ** 1.5)
        return min(1.0, max(0.0, score))
