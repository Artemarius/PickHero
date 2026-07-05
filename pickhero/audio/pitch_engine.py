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
- "experimental_ml": uses PitchEngine with optional ML assist (Step 8).

The audio callback must NEVER block on the worker. If the ring buffer is full,
the oldest buffer is dropped (xrun logged).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from pickhero.audio.detector import PitchDetector, DetectedNote
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
    source_flags: set[str] = field(default_factory=set)  # {"yin", "yin_4096", "spectral", "tab_prior", "octave_corrected"}

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

    # Rolling spectral buffer size for the validator.
    _SPECTRAL_BUF_SIZE = 4096

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
        self._ml_model_path = ml_model_path

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

        # ML assist (Step 8 — None unless experimental_ml profile)
        self._ml_assist = None
        if profile == "experimental_ml":
            self._init_ml_assist()

    def _init_ml_assist(self) -> None:
        """Try to load ML assist. Falls back to None if unavailable."""
        try:
            from pickhero.audio.ml_assist import MLAssist
            if self._ml_model_path:
                model_path = self._ml_model_path
            else:
                from pathlib import Path
                model_path = str(Path.home() / ".pickhero" / "models" / "crepe_small.onnx")
            self._ml_assist = MLAssist(model_path=model_path)
            if not self._ml_assist.available:
                self._ml_assist = None
        except Exception:
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
        """Run the consensus pipeline on a single hop-sized chunk.

        Calls PitchDetector.process() ONCE — which internally calls both
        aubio.pitch and aubio.onset. The is_onset and onset_sample come
        from that single call. The worker never calls _onset() again.
        """
        # Push to the rolling spectral buffer BEFORE processing (so the
        # spectral check has the latest audio available).
        self._push_spectral(chunk)

        # Step 1: Primary YIN + onset (called ONCE via detector.process)
        result = self._detector.process(chunk)
        if result is None:
            return None

        freq = result.frequency
        midi = result.midi_note
        confidence = result.confidence
        is_onset = result.is_onset
        onset_sample = result.onset_sample
        flags = {"yin"}

        if freq <= 0 or midi <= 0:
            return _EngineResult(
                candidate=PitchCandidate(
                    best_midi=None,
                    cents_error=None,
                    raw_frequency=freq,
                    confidence=confidence,
                    source_flags=flags,
                ),
                is_onset=is_onset,
                onset_sample=onset_sample,
                chunk_start_sample=0,
            )

        # Step 2: Multi-resolution — run second YIN at larger window for low notes.
        # Only contributes if the large detector has higher confidence.
        if self._detector_large is not None and freq < 200.0:
            try:
                result_large = self._detector_large.process(chunk)
                if result_large is not None and result_large.confidence > confidence:
                    freq = result_large.frequency
                    midi = result_large.midi_note
                    confidence = result_large.confidence
                    flags.add("yin_4096")
                    # Use the larger detector's onset if the primary didn't fire
                    if not is_onset and result_large.is_onset:
                        is_onset = True
                        onset_sample = result_large.onset_sample
            except Exception:
                pass

        # Step 3: Spectral sanity check on the rolling 4096 buffer (not the hop chunk).
        # The hop chunk at 256 samples gives 187.5 Hz bins — useless for low notes.
        # The 4096 rolling buffer gives 11.7 Hz bins, enough to validate E2 (82 Hz).
        spectral_ok = self._spectral_check(freq)
        if spectral_ok:
            flags.add("spectral")

        # Step 4: Tab prior — adjust confidence but never override the pitch
        with self._tab_prior_lock:
            tab_prior = set(self._tab_prior)
        if midi in tab_prior:
            flags.add("tab_prior")
            confidence = min(1.0, confidence + 0.05)

        # Step 5: Cents error
        nearest_midi, cents = freq_to_cents_deviation(freq)

        # Step 6: Combine confidence
        if not spectral_ok:
            confidence *= 0.7

        candidate = PitchCandidate(
            best_midi=midi,
            cents_error=cents,
            raw_frequency=freq,
            confidence=confidence,
            source_flags=flags,
        )
        return _EngineResult(
            candidate=candidate,
            is_onset=is_onset,
            onset_sample=onset_sample,
            chunk_start_sample=0,
        )

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

    def _spectral_check(self, freq: float) -> bool:
        """Harmonic-product sanity check on the rolling 4096-sample buffer.

        The rolling buffer gives ~11.7 Hz bin spacing at 48 kHz — enough to
        validate low guitar notes (E2 = 82 Hz). The hop-sized FFT (187.5 Hz
        bins) was useless for this.
        """
        if freq <= 0 or self._spectral_fill < self._SPECTRAL_BUF_SIZE // 2:
            return False

        buf = self._spectral_buf[:self._SPECTRAL_BUF_SIZE] * self._spectral_window
        spectrum = np.abs(np.fft.rfft(buf))

        # Find the bin closest to the detected frequency
        idx = np.argmin(np.abs(self._spectral_freqs - freq))
        if idx >= len(spectrum):
            return False

        # Check energy at the fundamental and 2nd harmonic
        fundamental_energy = spectrum[idx]
        idx_2h = np.argmin(np.abs(self._spectral_freqs - 2 * freq))
        harmonic_energy = spectrum[idx_2h] if idx_2h < len(spectrum) else 0.0

        total_energy = float(np.sum(spectrum))
        if total_energy <= 0:
            return False

        ratio = max(fundamental_energy, harmonic_energy) / total_energy
        return ratio > 0.005  # lenient threshold — this is a sanity check
