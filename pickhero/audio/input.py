"""Audio input capture using sounddevice.

Runs a sounddevice InputStream that feeds audio buffers to the pitch detector.
Detected notes are pushed to a thread-safe queue for consumption by the main thread.
"""

import queue
import threading
import time
import sys
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from pickhero.audio.chord_detector import ChordDetector
from pickhero.audio.detector import PitchDetector, DetectedNote
from pickhero.audio.performance import PerformanceEvent
from pickhero.config import Config

@dataclass
class TimestampedNote:
    """A detected note with a timestamp (ms from session start)."""
    note: DetectedNote
    timestamp_ms: float


class AudioCapture:
    """Captures audio from an input device and runs pitch detection.

    Detected notes are pushed to `note_queue` for consumption by other threads.
    The sounddevice callback runs in a separate thread automatically.
    """

    def __init__(self, config: Config | None = None):
        if config is None:
            config = Config()
        self.config = config
        ac = config.audio

        calibration = getattr(config, 'calibration', None) or None

        # Select pitch engine based on accuracy profile.
        # "portable" uses PitchDetector directly (existing behavior).
        # "high_accuracy"/"experimental_ml" uses PitchEngine with a worker thread.
        # CRITICAL: the stream, detector, chord_detector, and engine must ALL
        # use the same sample rate. We update ac.sample_rate in-place so the
        # stream opened in start() uses the same rate as the detector.
        self._profile = ac.profile
        if ac.profile in ("high_accuracy", "experimental_ml"):
            from pickhero.audio.pitch_engine import PitchEngine
            if ac.profile == "high_accuracy":
                # Request 48 kHz, hop 256, buf 4096. Update ac so the stream
                # and chord_detector use the same rate — no split-brain.
                ac.sample_rate = max(ac.sample_rate, 48000)
                ac.hop_size = min(ac.hop_size, 256)
                ac.buf_size = max(ac.buf_size, 4096)
            self._engine = PitchEngine(
                sample_rate=ac.sample_rate,
                hop_size=ac.hop_size,
                buf_size=ac.buf_size,
                confidence_threshold=ac.confidence_threshold,
                onset_threshold=ac.onset_threshold,
                noise_gate_db=ac.noise_gate_db,
                calibration=calibration if calibration else None,
                profile=ac.profile,
                ml_model_path=ac.ml_model_path,
            )
            self.detector = self._engine.detector
        else:
            self._engine = None
            self.detector = PitchDetector(
                buf_size=ac.buf_size,
                hop_size=ac.hop_size,
                sample_rate=ac.sample_rate,
                confidence_threshold=ac.confidence_threshold,
                onset_threshold=ac.onset_threshold,
                noise_gate_db=ac.noise_gate_db,
                calibration=calibration if calibration else None,
            )
        self.note_queue: queue.Queue[TimestampedNote] = queue.Queue()
        self.event_queue: queue.Queue[PerformanceEvent] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._start_time: float = 0.0
        self._signal_db: float = -120.0
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self.chord_detector = ChordDetector(sample_rate=ac.sample_rate)
        # Absolute sample offset consumed by the detector (advances by hop per process() call)
        self._detector_sample_offset: int = 0
        self._adc_time_available: bool | None = None
        self._xrun_count: int = 0
        # Patch 6b: opt-in raw take-audio ring buffer for offline polyphonic
        # analysis. None unless start_take_recording() is called.
        self._take_audio: list[np.ndarray] | None = None

        # Track stabilizer: multi-frame consensus before notes reach the
        # matcher. Raw per-frame detections go through here; only stable
        # note events emerge. Prevents octave glitches, noise transients,
        # and sustain-frame spam from reaching the matcher.
        from pickhero.audio.track_stabilizer import TrackStabilizer
        self._stabilizer = TrackStabilizer(
            sample_rate=ac.sample_rate, hop_size=ac.hop_size,
        )
        # Tab context: expected MIDI notes near the current playback position.
        # Fed to the stabilizer for octave resolution. Set by the playback
        # loop (scrolling.py:update) via set_tab_context().
        self._tab_expected_midi: list[int] = []
        self._tab_context_ms: float = 0.0
        # _worker_in_queue.  Replaces the per-hop DSP that used to live in
        # _audio_callback for both portable and high_accuracy profiles.
        self._worker_in_queue: queue.Queue = queue.Queue(maxsize=256)
        self._worker_thread: threading.Thread | None = None
        self._worker_running: bool = False
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Sounddevice callback — runs in audio thread.

        This method is real-time safe: it performs only cheap, non-blocking
        operations (copy, append, queue.put).  All DSP runs on the unified
        worker thread.
        """
        if status:
            if "input overflow" in str(status).lower():
                self._xrun_count += 1
            elif "input underflow" in str(status).lower() or "output" in str(status).lower():
                self._xrun_count += 1
                return
            else:
                self._xrun_count += 1

        # indata shape: (frames, channels) — take first channel
        mono = indata[:, 0].copy()

        # Patch 6b: record raw mono audio for offline analysis when armed.
        if self._take_audio is not None:
            self._take_audio.append(mono.copy())

        # Feed chord detector (FFT-based, runs on full buffer)
        self.chord_detector.push_audio(mono)

        # Determine the ADC timestamp of this buffer's first sample.
        adc_time = getattr(time_info, "inputBufferAdcTime", 0.0) if time_info else 0.0
        if self._adc_time_available is None:
            self._adc_time_available = adc_time > 0.0
            if self._adc_time_available:
                self._start_time = adc_time
            else:
                self._start_time = time.perf_counter()
        elif self._adc_time_available and self._start_time == 0.0:
            self._start_time = adc_time

        sample_rate = self.detector.sample_rate
        hop = self.detector.hop_size

        # Push hop-sized chunks into the worker input queue with timestamps.
        for i in range(0, len(mono) - hop + 1, hop):
            chunk = mono[i:i + hop]
            self._worker_in_queue.put((chunk, self._detector_sample_offset))
            self._detector_sample_offset += hop

        # Drain completed PerformanceEvents from the articulation detector.
        for event in self.detector.drain_events():
            self.event_queue.put(event)
    def _start_unified_worker(self) -> None:
        """Start the unified worker thread that processes queued audio chunks."""
        if self._worker_running:
            return
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._unified_worker_loop, daemon=True
        )
        self._worker_thread.start()

    def _unified_worker_loop(self) -> None:
        """Worker thread: drain chunks, process, emit stable events.

        For high_accuracy profiles the worker also interacts with the
        PitchEngine.  For portable profiles it calls PitchDetector.process
        directly.  Results flow through _emit_through_stabilizer → note_queue.

        CRITICAL: we only emit through the stabilizer when a real chunk
        was processed.  Idle gaps (queue.Empty) are simply skipped — they
        must not age out stabilizer tracks because nothing is happening yet.
        """
        sample_rate = self.detector.sample_rate

        while self._worker_running:
            try:
                chunk, chunk_start_sample = self._worker_in_queue.get(timeout=0.05)
            except queue.Empty:
                # Idle gap — do NOT emit None to the stabilizer.  Emitting
                # None every 50 ms ages out active tracks before multi-frame
                # consensus can form.  We only process chunks that actually
                # arrived.
                continue

            if self._engine is not None:
                self._engine.submit(chunk, chunk_start_sample)
                self._signal_db = self.detector.last_signal_db
                self._tuner_freq = self.detector.last_freq
                self._tuner_confidence = self.detector.last_confidence
                frame_ms = chunk_start_sample / sample_rate * 1000.0
                had_result = False
                for r in self._engine.get_candidates():
                    had_result = True
                    result = r.candidate.to_detected_note(
                        is_onset=r.is_onset, onset_sample=r.onset_sample,
                        performance=r.performance,
                    )
                    if result is not None:
                        self._emit_through_stabilizer(result, frame_ms)
                if not had_result:
                    self._emit_through_stabilizer(None, frame_ms)
            else:
                result = self.detector.process(chunk)
                self._signal_db = self.detector.last_signal_db
                self._tuner_freq = self.detector.last_freq
                self._tuner_confidence = self.detector.last_confidence
                frame_ms = chunk_start_sample / sample_rate * 1000.0
                if result is not None:
                    self._emit_through_stabilizer(result, frame_ms)
                else:
                    self._emit_through_stabilizer(None, frame_ms)
    def _compute_timestamp_ms(
        self,
        result: DetectedNote,
        chunk_offset_in_buffer: int,
        adc_time: float,
        sample_rate: int,
        hop: int,
    ) -> float:
        """Compute timestamp (ms from session start) from the onset's absolute sample position.

        The absolute sample index is the canonical timestamp. The ADC time is used
        only to anchor the sample clock to wall-clock for the first buffer; after
        that the system advances strictly by sample offsets.
        """
        if not self._adc_time_available:
            # Backend without ADC time — use wall clock (legacy behavior)
            return (time.perf_counter() - self._start_time) * 1000.0

        onset_sample = result.onset_sample
        # onset_sample is absolute (delay-compensated) since detector creation.
        # If unavailable (non-onset path), fall back to the current chunk's stream position.
        if onset_sample is None:
            onset_sample = self._detector_sample_offset
        elif onset_sample < 0:
            # Start-of-file edge in aubio: clamp to 0 (the stream beginning).
            onset_sample = 0

        return max(0.0, onset_sample) / sample_rate * 1000.0

    def set_tab_context(self, expected_midi: list[int], current_ms: float) -> None:
        """Set the expected tab notes near the current playback position.

        Called by the playback loop (scrolling.py:update) every frame.
        The stabilizer uses this as a prior for octave resolution.
        """
        self._tab_expected_midi = expected_midi
        self._tab_context_ms = current_ms

    def _emit_through_stabilizer(self, result: DetectedNote, ts_ms: float) -> None:
        """Feed a raw detection through the track stabilizer.

        Only stable note events (multi-frame consensus) reach the note queue.
        """
        tab_prior = None
        if self._tab_expected_midi:
            tab_prior = min(
                self._tab_expected_midi,
                key=lambda m: abs(m - result.midi_note) if result.midi_note > 0 else abs(m - 60),
            )
        events = self._stabilizer.process(result, ts_ms, tab_prior_midi=tab_prior)
        for event in events:
            stable_note = DetectedNote(
                midi_note=event.midi_note,
                frequency=event.frequency,
                confidence=event.confidence,
                name=event.name,
                is_onset=event.is_onset,
                onset_sample=event.onset_sample,
                performance=event.performance,
                event_snapshot=event.event_snapshot,
            )
            self.note_queue.put(
                TimestampedNote(note=stable_note, timestamp_ms=event.timestamp_ms)
            )

    def _resolve_device(self) -> int | None:
        """Resolve device_name to a current index, preferring mono inputs.

        Falls back to device_index if name doesn't match.
        Updates sample_rate to match the resolved device's default.
        """
        ac = self.config.audio
        if ac.device_name:
            try:
                devices = sd.query_devices()
                matches = []
                for i, dev in enumerate(devices):
                    if dev["max_input_channels"] > 0 and ac.device_name in dev["name"]:
                        matches.append((i, dev))
                if matches:
                    # Prefer 1-channel devices (split/mono inputs), then fewest channels
                    matches.sort(key=lambda x: x[1]["max_input_channels"])
                    best_idx, best_dev = matches[0]
                    # Update sample rate to match device default
                    default_sr = int(best_dev["default_samplerate"])
                    if default_sr > 0:
                        ac.sample_rate = default_sr
                    return best_idx
            except Exception:
                pass
        return ac.device_index

    def start(self):
        """Start audio capture.

        Resolves the device by name first, then recreates the detector
        with the correct sample rate before opening the stream.
        """
        ac = self.config.audio

        # Resolve device name → index, updating sample_rate to match device default
        resolved = self._resolve_device()

        # Recreate detector with the resolved sample rate (may have changed)
        calibration = getattr(self.config, 'calibration', None) or None
        if self._profile in ("high_accuracy", "experimental_ml"):
            from pickhero.audio.pitch_engine import PitchEngine
            # CRITICAL: use the SAME sample rate as the stream (ac.sample_rate).
            # The init already updated ac.sample_rate for high_accuracy; we must
            # not override it here. No split-brain: stream == detector == engine.
            self._engine = PitchEngine(
                sample_rate=ac.sample_rate,
                hop_size=ac.hop_size,
                buf_size=ac.buf_size,
                confidence_threshold=ac.confidence_threshold,
                onset_threshold=ac.onset_threshold,
                noise_gate_db=ac.noise_gate_db,
                calibration=calibration if calibration else None,
                profile=self._profile,
                ml_model_path=ac.ml_model_path,
            )
            self.detector = self._engine.detector
        else:
            self._engine = None
            from pickhero.audio.detector import PitchDetector
            self.detector = PitchDetector(
                buf_size=ac.buf_size,
                hop_size=ac.hop_size,
                sample_rate=ac.sample_rate,
                confidence_threshold=ac.confidence_threshold,
                onset_threshold=ac.onset_threshold,
                noise_gate_db=ac.noise_gate_db,
                calibration=calibration if calibration else None,
            )
        self.detector.reset()
        self.chord_detector.reset()
        self.chord_detector.set_sample_rate(ac.sample_rate)
        # Drain any leftover notes
        while not self.note_queue.empty():
            try:
                self.note_queue.get_nowait()
            except queue.Empty:
                break

        # Reset timing state — detector was recreated, so sample offsets restart at 0
        self._detector_sample_offset = 0
        self._adc_time_available = None
        self._xrun_count = 0
        # _start_time will be set on first callback from time_info.inputBufferAdcTime
        self._start_time = 0.0
        # Start the unified worker thread
        self._start_unified_worker()
        # Low-latency mode: uses default_low_input_latency (~9ms vs ~35ms default_high).
        # On Windows, request WASAPI exclusive mode for ~3ms hardware latency.
        extra = None
        if sys.platform == 'win32':
            try:
                extra = sd.WasapiSettings(exclusive=True)
            except AttributeError:
                # Older sounddevice without WasapiSettings — shared mode only.
                pass
        try:
            self._stream = sd.InputStream(
                device=resolved,
                channels=1,
                samplerate=ac.sample_rate,
                blocksize=ac.hop_size,
                dtype="float32",
                latency='low',
                extra_settings=extra,
                callback=self._audio_callback,
            )
        except sd.PortAudioError:
            # Device doesn't support latency='low' — retry with default latency
            self._stream = sd.InputStream(
                device=resolved,
                channels=1,
                samplerate=ac.sample_rate,
                blocksize=ac.hop_size,
                dtype="float32",
                extra_settings=extra,
                callback=self._audio_callback,
            )
        self._stream.start()
        if self._engine is not None:
            self._engine.start()

    def stop(self):
        """Stop audio capture."""
        if self._engine is not None:
            self._engine.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Stop the unified worker thread
        self._worker_running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        # Flush any pending track that reached consensus but wasn't emitted
        # during the last callback (e.g., a note still ringing at stop time).
        for event in self._stabilizer.flush():
            stable_note = DetectedNote(
                midi_note=event.midi_note,
                frequency=event.frequency,
                confidence=event.confidence,
                name=event.name,
                is_onset=event.is_onset,
                onset_sample=event.onset_sample,
                performance=event.performance,
                event_snapshot=event.event_snapshot,
            )
            self.note_queue.put(
                TimestampedNote(note=stable_note, timestamp_ms=event.timestamp_ms)
            )

    def set_noise_gate_db(self, db: float) -> None:
        """Update the noise gate threshold on the detector.

        Thread-safe: single float attribute write is atomic under the GIL.
        """
        self.detector.set_noise_gate_db(db)

    def get_signal_db(self) -> float:
        """Return the latest signal level in dB. Thread-safe (single float read under GIL)."""
        return self._signal_db

    def get_tuner_data(self) -> tuple[float, float]:
        """Return (frequency_hz, confidence) for tuner display. Thread-safe."""
        return (self._tuner_freq, self._tuner_confidence)

    def get_xrun_count(self) -> int:
        """Return the number of buffer overflows since start. Thread-safe."""
        return self._xrun_count

    def get_notes(self) -> list[TimestampedNote]:
        """Drain all pending detected notes from the queue (non-blocking)."""
        notes = []
        while True:
            try:
                notes.append(self.note_queue.get_nowait())
            except queue.Empty:
                break
        return notes

    def get_events(self) -> list[PerformanceEvent]:
        """Drain all pending PerformanceEvents from the queue (non-blocking).

        A PerformanceEvent is pushed when a note is closed (next onset fires).
        The matcher pairs these with NoteEvents for the after-take analyzer.
        """
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def start_take_recording(self) -> None:
        """Arm the raw-audio ring buffer for offline polyphonic analysis.

        Subsequent audio callbacks append mono chunks until
        :meth:`stop_take_recording` is called. No-op if already recording.
        """
        if self._take_audio is None:
            self._take_audio = []

    def stop_take_recording(self) -> np.ndarray | None:
        """Stop recording and return the concatenated raw mono audio.

        Returns None if recording was never armed. The buffer is cleared after
        reading so the memory is released.
        """
        if self._take_audio is None:
            return None
        chunks = self._take_audio
        self._take_audio = None
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

def list_audio_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            api_name = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else "?"
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
                "hostapi": api_name,
            })
    return inputs


def validate_device_index(index: int | None) -> bool:
    """Check if a device index exists and has input channels.

    Returns True for None (system default) or a valid input device index.
    """
    if index is None:
        return True
    try:
        info = sd.query_devices(index)
        return info["max_input_channels"] > 0
    except (sd.PortAudioError, IndexError, ValueError):
        return False

