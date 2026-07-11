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
from pickhero.audio.detector import PitchDetector, DetectedNote
from pickhero.audio.match_mode import MatchMode, _coerce_match_mode
from pickhero.audio.performance import PerformanceEvent
from pickhero.config import Config, LatencyBreakdown
from pickhero.audio.drift_monitor import DriftMonitor
from pickhero.audio.clock import StreamClock

@dataclass
class TimestampedNote:
    """A detected note with a timestamp (ms from session start)."""
    note: DetectedNote
    timestamp_ms: float


@dataclass
class _AudioBlock:
    """One immutable callback block handed to the DSP worker."""

    samples: np.ndarray
    start_sample: int


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
        # "high_accuracy" uses PitchEngine with a worker thread.
        # CRITICAL: the stream, detector, and engine must all
        # use the same sample rate. We update ac.sample_rate in-place so the
        # stream opened in start() uses the same rate as the detector.
        self._profile = ac.profile
        if ac.profile == "high_accuracy":
            from pickhero.audio.pitch_engine import PitchEngine
            # Request 48 kHz, hop 256, buf 4096. Update ac so the stream
            # and stream use the same rate — no split-brain.
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
        self._input_peak: float = 0.0
        self._clipped_fraction: float = 0.0
        self._dc_offset: float = 0.0
        self._input_channel: int = max(0, int(ac.input_channel))
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self.chord_detector = None  # compatibility attribute; live scorer no longer uses it
        # Absolute number of input samples observed by the callback. Worker block
        # timestamps use this capture-domain counter and explicitly reset at gaps.
        self._detector_sample_offset: int = 0
        # Callback-side hop framing: incoming audio is split into hop-sized
        # chunks before enqueuing. The tail carries over to the next callback.
        self._callback_hop_size: int = ac.hop_size
        # Preallocated buffers for zero-allocation audio callback.
        # Double-buffered combined buffer: callback writes to one while
        # the worker reads from the other via _AudioBlock views.
        self._callback_combined = [np.zeros(ac.hop_size * 16, dtype=np.float32) for _ in range(2)]
        self._callback_remainder = np.zeros(ac.hop_size, dtype=np.float32)
        self._callback_buffer_idx: int = 0
        self._callback_carry: np.ndarray = self._callback_remainder[:0]
        self._adc_time_available: bool | None = None
        self._xrun_count: int = 0
        # Patch 6b: opt-in raw take-audio ring buffer for offline polyphonic
        # analysis. None unless start_take_recording() is called.
        self._take_audio: list[np.ndarray] | None = None

        # Raw-audio ring buffer for expected-event verification (~4 seconds).
        # Written by the audio callback and read by get_recent_audio().
        self._RING_DURATION_MS = 4000.0
        ring_samples = int(ac.sample_rate * self._RING_DURATION_MS / 1000.0)
        self._audio_ring: np.ndarray = np.zeros(ring_samples, dtype=np.float32)
        # Absolute number of samples captured into the ring. This is separate
        # from the detector worker offset: worker queue pressure must never
        # change the raw-audio clock used by verifier windows.
        self._ring_write_sample: int = 0
        self._ring_write_idx: int = 0
        self._ring_sample_rate: int = ac.sample_rate
        # Even = stable snapshot; odd = callback is writing. Readers copy and
        # validate the sequence rather than blocking the real-time callback.
        self._ring_sequence: int = 0
        self.clock = StreamClock(
            latency_offset_ms=config.get_audio_latency_offset()
        )

        # Track stabilizer: multi-frame consensus before notes reach the
        # matcher. Raw per-frame detections go through here; only stable note
        # events emerge, preventing octave glitches and noise transients.
        import warnings
        from pickhero.audio.track_stabilizer import TrackStabilizer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self._stabilizer = TrackStabilizer(
                sample_rate=ac.sample_rate,
                hop_size=ac.hop_size,
                mode=_coerce_match_mode(config.match_mode),
            )
        # Tab context: expected MIDI notes near the current playback position.
        # Fed to the stabilizer for octave resolution. Set by the playback
        # loop (scrolling.py:update) via set_tab_context().
        self._tab_expected_midi: list[int] = []
        self._tab_context_ms: float = 0.0
        # The callback enqueues one block, not one queue item per hop. All
        # telemetry, chord analysis, take recording and detector DSP run on the
        # worker. The verification ring remains callback-owned so raw scoring
        # audio is never delayed behind detector work.
        self._worker_in_queue: queue.Queue[_AudioBlock] = queue.Queue(maxsize=128)
        self._worker_thread: threading.Thread | None = None
        self._worker_running: bool = False
        self._worker_carry = np.zeros(0, dtype=np.float32)
        self._worker_carry_start_sample: int = 0
        self._worker_expected_sample: int | None = None
        self._worker_dropped_samples: int = 0
        # Clock drift monitor — tracks ADC vs sample-count divergence.
        self._drift_monitor = DriftMonitor(sample_rate=ac.sample_rate)
        self._reported_input_latency_ms: float = 0.0
        self._reported_output_latency_ms: float = 0.0

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """PortAudio callback with bounded, non-blocking work.

        Splits incoming audio into hop-sized chunks, enqueues each to the
        worker, and appends to take recording when armed.
        """
        if status:
            self._xrun_count += 1
        channel = min(self._input_channel, max(0, indata.shape[1] - 1))
        mono_size = indata.shape[0]
        if mono_size == 0:
            return

        # Use indata[:, channel] directly — it's a view (zero-allocation).
        # Within the callback we copy data into the ring and combined buffers,
        # so PortAudio reusing indata after return is safe.
        mono = indata[:, channel]

        # The raw verifier clock is callback-owned and therefore reflects every
        # captured sample even when the detector worker falls behind.
        self._write_audio_ring(mono)

        adc_time = getattr(time_info, "inputBufferAdcTime", 0.0) if time_info else 0.0
        if self._adc_time_available is None:
            self._adc_time_available = adc_time > 0.0
            self._start_time = adc_time if self._adc_time_available else time.perf_counter()
        elif self._adc_time_available and self._start_time == 0.0:
            self._start_time = adc_time

        self._drift_monitor.update(adc_time, self._ring_write_sample)

        # Split into hop-sized chunks using preallocated double-buffered
        # combined buffer. Alternate between slot 0 and 1 so the worker
        # can safely read from the previous slot while we write here.
        hop = self._callback_hop_size
        buf_idx = self._callback_buffer_idx
        self._callback_buffer_idx = 1 - buf_idx
        combined_buf = self._callback_combined[buf_idx]
        carry = self._callback_carry
        carry_size = carry.size
        total_size = carry_size + mono_size
        combined_buf[:carry_size] = carry
        combined_buf[carry_size:total_size] = mono
        combined = combined_buf[:total_size]

        offset = 0
        while offset + hop <= combined.size:
            chunk = combined[offset:offset + hop]
            block_start = self._detector_sample_offset + offset
            block = _AudioBlock(samples=chunk, start_sample=block_start)
            try:
                self._worker_in_queue.put_nowait(block)
            except queue.Full:
                # Prefer fresh audio over stale detector work. Drop oldest.
                try:
                    dropped = self._worker_in_queue.get_nowait()
                    self._worker_dropped_samples += int(dropped.samples.size)
                    self._worker_in_queue.put_nowait(block)
                except (queue.Empty, queue.Full):
                    self._worker_dropped_samples += int(chunk.size)
                self._xrun_count += 1
            # Take recording happens per-hop in the callback.
            recording = self._take_audio
            if recording is not None:
                recording.append(chunk)
            offset += hop

        self._detector_sample_offset += int(mono_size)
        # Retain the remainder for the next callback (no-alloc copy).
        if offset < combined.size:
            remainder_size = combined.size - offset
            self._callback_remainder[:remainder_size] = combined[offset:]
            self._callback_carry = self._callback_remainder[:remainder_size]
        else:
            self._callback_carry = self._callback_remainder[:0]

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
        """Drain callback blocks and run every non-real-time operation."""
        sample_rate = self.detector.sample_rate
        hop = self.detector.hop_size

        while self._worker_running:
            try:
                block = self._worker_in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            samples = block.samples
            if samples.size == 0:
                continue

            # Interface health telemetry belongs here, not in the PortAudio
            # callback. These reductions may allocate and are not deadline-safe.
            peak = float(np.max(np.abs(samples)))
            clipped = float(np.mean(np.abs(samples) >= 0.985))
            dc = float(np.mean(samples))
            self._input_peak = self._input_peak * 0.72 + peak * 0.28
            self._clipped_fraction = self._clipped_fraction * 0.82 + clipped * 0.18
            self._dc_offset = self._dc_offset * 0.90 + dc * 0.10


            if (
                self._worker_expected_sample is not None
                and block.start_sample != self._worker_expected_sample
            ):
                # A queue overrun created a discontinuity. Never concatenate the
                # old tail with new audio or let pitch consensus bridge the gap.
                self._worker_carry = np.zeros(0, dtype=np.float32)
                self._worker_carry_start_sample = block.start_sample
                self._stabilizer.reset()
                if self._engine is not None:
                    self._engine.reset()
                else:
                    self.detector.reset()

            if self._worker_carry.size:
                combined = np.concatenate((self._worker_carry, samples))
                combined_start = self._worker_carry_start_sample
            else:
                combined = samples
                combined_start = block.start_sample

            complete = (combined.size // hop) * hop
            for offset in range(0, complete, hop):
                chunk = combined[offset:offset + hop]
                chunk_start_sample = combined_start + offset
                frame_ms = chunk_start_sample / sample_rate * 1000.0

                if self._engine is not None:
                    engine_result = self._engine.process_sync(chunk, chunk_start_sample)
                    self._signal_db = self.detector.last_signal_db
                    self._tuner_freq = self.detector.last_freq
                    self._tuner_confidence = self.detector.last_confidence
                    if engine_result is not None:
                        note = engine_result.candidate.to_detected_note(
                            is_onset=engine_result.is_onset,
                            onset_sample=engine_result.onset_sample,
                            performance=engine_result.performance,
                        )
                        self._emit_through_stabilizer(note, frame_ms)
                    else:
                        self._emit_through_stabilizer(None, frame_ms)
                else:
                    detected = self.detector.process(chunk)
                    self._signal_db = self.detector.last_signal_db
                    self._tuner_freq = self.detector.last_freq
                    self._tuner_confidence = self.detector.last_confidence
                    self._emit_through_stabilizer(detected, frame_ms)

            remainder = combined.size - complete
            if remainder:
                self._worker_carry = combined[complete:].copy()
                self._worker_carry_start_sample = combined_start + complete
            else:
                self._worker_carry = np.zeros(0, dtype=np.float32)
                self._worker_carry_start_sample = block.start_sample + samples.size
            self._worker_expected_sample = block.start_sample + samples.size

            for event in self.detector.drain_events():
                self.event_queue.put(event)

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

    def set_tab_context(
        self,
        expected_midi: list[int],
        current_ms: float,
        expected_techniques: set[str] | None = None,
    ) -> None:
        """Set the expected tab notes and techniques near the current position.

        Called by the playback loop (scrolling.py:update) every frame.
        The stabilizer uses this as a prior for octave resolution.
        """
        self._tab_expected_midi = expected_midi
        self._tab_context_ms = current_ms
        if self._engine is not None:
            self._engine.set_tab_prior(set(expected_midi))
        techniques = expected_techniques or set()
        self.detector.set_expected_techniques(techniques)

    def set_match_mode(self, mode: MatchMode | str) -> None:
        """Update the matching mode mid-session (e.g. UI mode toggle).

        Propagates the change to the track stabilizer so real-time
        consensus policies stay in sync with the matcher.
        """
        self._stabilizer.set_mode(mode)

    def _emit_through_stabilizer(self, result: DetectedNote | None, ts_ms: float) -> None:
        """Feed a raw detection through the track stabilizer.

        Only stable note events (multi-frame consensus) reach the note queue.
        """
        tab_prior = None
        if result is not None and self._tab_expected_midi:
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


    def list_asio_devices(self) -> list[dict]:
        """List available ASIO audio input devices.

        Returns a list of dicts with index, name, channels, sample_rate,
        and hostapi for each ASIO-capable input device.
        """
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        asio_devices = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                api_name = hostapis[dev["hostapi"]]["name"] if dev["hostapi"] < len(hostapis) else "?"
                if "ASIO" in api_name:
                    asio_devices.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "sample_rate": dev["default_samplerate"],
                        "hostapi": api_name,
                    })
        return asio_devices

    def start(self):
        """Start audio capture.

        Resolves the device by name first, then recreates the detector
        with the correct sample rate before opening the stream.
        """
        ac = self.config.audio

        # Resolve device name → index, updating sample_rate to match device default
        resolved = self._resolve_device()
        try:
            device_info = sd.query_devices(resolved, "input")
            max_channels = max(1, int(device_info["max_input_channels"]))
        except Exception:
            max_channels = 1
        self._input_channel = min(max(0, int(ac.input_channel)), max_channels - 1)
        ac.input_channel = self._input_channel
        stream_channels = self._input_channel + 1

        # Recreate detector with the resolved sample rate (may have changed)
        calibration = getattr(self.config, 'calibration', None) or None
        if self._profile == "high_accuracy":
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
        # Re-allocate the verification ring for the resolved sample rate.
        self._ring_sequence += 1
        try:
            ring_samples = int(ac.sample_rate * self._RING_DURATION_MS / 1000.0)
            self._audio_ring = np.zeros(ring_samples, dtype=np.float32)
            self._ring_write_sample = 0
            self._ring_write_idx = 0
            self._ring_sample_rate = ac.sample_rate
        finally:
            self._ring_sequence += 1
        self.clock.latency_offset_ms = self.config.get_audio_latency_offset()
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
        self._worker_dropped_samples = 0
        self._worker_carry = np.zeros(0, dtype=np.float32)
        self._worker_carry_start_sample = 0
        self._worker_expected_sample = None
        self._callback_buffer_idx = 0
        self._callback_carry = self._callback_remainder[:0]
        self._input_peak = 0.0
        self._clipped_fraction = 0.0
        self._dc_offset = 0.0
        # _start_time will be set on first callback from time_info.inputBufferAdcTime
        self._start_time = 0.0
        self._drift_monitor.reset(ac.sample_rate)
        # Start the unified worker thread
        self._start_unified_worker()
        # Low-latency mode: uses default_low_input_latency (~9ms vs ~35ms default_high).
        # On Windows, request ASIO exclusive (if enabled), then WASAPI exclusive, then shared.
        extra = None
        blocksize = ac.hop_size
        if sys.platform == 'win32':
            if ac.asio_enabled:
                try:
                    extra = sd.AsioSettings(channel_selectors=[ac.input_channel])
                except AttributeError:
                    # Older sounddevice without AsioSettings — skip to WASAPI.
                    pass
            if extra is None:
                try:
                    extra = sd.WasapiSettings(exclusive=True)
                except AttributeError:
                    # Older sounddevice without WasapiSettings — shared mode only.
                    pass
        if ac.asio_buffer_size > 0 and ac.asio_enabled:
            blocksize = ac.asio_buffer_size
        latency_map = {"low": "low", "medium": "low", "high": "high"}
        requested_latency = latency_map.get(ac.latency_mode, "low")
        _channels = 1 if isinstance(extra, sd.AsioSettings) else stream_channels
        try:
            self._stream = sd.InputStream(
                device=resolved,
                channels=_channels,
                samplerate=ac.sample_rate,
                blocksize=blocksize,
                dtype="float32",
                latency=requested_latency,
                extra_settings=extra,
                callback=self._audio_callback,
            )
        except sd.PortAudioError:
            # ASIO failed → try WASAPI exclusive, then default.
            if isinstance(extra, sd.AsioSettings):
                try:
                    extra = sd.WasapiSettings(exclusive=True)
                except AttributeError:
                    extra = None
                blocksize = ac.hop_size
                try:
                    self._stream = sd.InputStream(
                        device=resolved,
                        channels=stream_channels,
                        samplerate=ac.sample_rate,
                        blocksize=blocksize,
                        dtype="float32",
                        latency="low",
                        extra_settings=extra,
                        callback=self._audio_callback,
                    )
                except sd.PortAudioError:
                    self._stream = sd.InputStream(
                        device=resolved,
                        channels=stream_channels,
                        samplerate=ac.sample_rate,
                        blocksize=ac.hop_size,
                        dtype="float32",
                        extra_settings=None,
                        callback=self._audio_callback,
                    )
            else:
                # Device does not support the requested host latency. Keep the
                # selected device/rate/block size and let PortAudio choose latency.
                self._stream = sd.InputStream(
                    device=resolved,
                    channels=stream_channels,
                    samplerate=ac.sample_rate,
                    blocksize=ac.hop_size,
                    dtype="float32",
                    extra_settings=extra,
                    callback=self._audio_callback,
                )
        self._stream.start()
        try:
            latency = self._stream.latency
            if isinstance(latency, (tuple, list)):
                self._reported_input_latency_ms = max(0.0, float(latency[0]) * 1000.0)
                self._reported_output_latency_ms = max(0.0, float(latency[-1]) * 1000.0) if len(latency) > 1 else 0.0
            else:
                self._reported_input_latency_ms = max(0.0, float(latency) * 1000.0)
                self._reported_output_latency_ms = 0.0
        except (AttributeError, TypeError, ValueError):
            self._reported_input_latency_ms = 0.0
            self._reported_output_latency_ms = 0.0
        self.clock.reset()
        # Engine worker thread removed — process_sync() runs on the
        # unified worker thread. No separate engine thread to start.

    def stop(self):
        """Stop audio capture."""
        # Engine worker thread removed — no separate engine thread to stop.
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Stop the unified worker thread
        self._worker_running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        # Drain any queued chunks that never got processed.
        while True:
            try:
                self._worker_in_queue.get_nowait()
            except queue.Empty:
                break
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

    def get_input_health(self) -> dict[str, float | int | bool | str]:
        """Return actionable interface-level diagnostics for the HUD."""
        clipping = self._input_peak >= 0.985 or self._clipped_fraction >= 0.002
        dc_problem = abs(self._dc_offset) >= 0.05
        if clipping:
            status = "clipping"
        elif dc_problem:
            status = "dc_offset"
        elif self._worker_dropped_samples > 0:
            status = "overrun"
        else:
            status = "ok"
        return {
            "status": status,
            "clipping": clipping,
            "dc_problem": dc_problem,
            "peak": self._input_peak,
            "clipped_fraction": self._clipped_fraction,
            "dc_offset": self._dc_offset,
            "channel": self._input_channel,
            "worker_backlog": self._worker_in_queue.qsize(),
            "dropped_samples": self._worker_dropped_samples,
            "reported_input_latency_ms": self._reported_input_latency_ms,
            "reported_output_latency_ms": self._reported_output_latency_ms,
        }

    def get_latency_breakdown(self) -> dict[str, float | bool]:
        """Return measured/reported latency components for diagnostics.

        ADC timestamps backdate captured samples on capable host APIs, so the
        reported device latency is informational in that case. On backends
        without ADC timestamps it is the best automatic baseline available.
        The persisted profile remains a user/loopback calibration trim.

        Returns both the new unified keys (``input_latency_ms``,
        ``output_latency_ms``, ``detector_window_ms``, …) and the legacy keys
        (``reported_input_ms``, ``onset_detector_ms``, …) for backward compat.
        """
        sample_rate = max(1, int(self.detector.sample_rate))
        try:
            onset_delay_ms = max(0.0, self.detector.get_onset_delay() / sample_rate * 1000.0)
        except Exception:
            onset_delay_ms = 0.0
        consensus_delay_ms = max(0.0, 2.0 * self.detector.hop_size / sample_rate * 1000.0)

        # Input / output latency from the stream (portaudio-reported).
        input_latency_ms = max(0.0, self._reported_input_latency_ms)
        output_latency_ms = max(0.0, self._reported_output_latency_ms)

        # Detector analysis window.
        #   Spectral mode (high-accuracy profile): rolling FFT buffer.
        #   YIN mode (portable profile): per-frame onset delay (algorithmic).
        if self._engine is not None:
            detector_window_ms = max(0.0, self._engine._SPECTRAL_BUF_SIZE / sample_rate * 1000.0)
        else:
            detector_window_ms = onset_delay_ms

        # Estimated display render budget (~60 fps → 16.67 ms).
        render_display_ms = 1000.0 / 60.0

        # Manual or auto-calibration trim.
        trim_ms = self.config.get_audio_latency_offset()

        # Total = sum of all contributing non-negative latencies.
        total_latency_ms = (
            input_latency_ms
            + output_latency_ms
            + detector_window_ms
            + consensus_delay_ms
            + render_display_ms
            + trim_ms
        )

        adc = bool(self._adc_time_available)

        result: dict[str, float | bool] = {
            # New unified keys
            "input_latency_ms": input_latency_ms,
            "output_latency_ms": output_latency_ms,
            "detector_window_ms": detector_window_ms,
            "stabilizer_confirmation_ms": consensus_delay_ms,
            "render_display_ms": render_display_ms,
            "manual_or_loopback_trim_ms": trim_ms,
            "total_latency_ms": total_latency_ms,
            "adc_timestamped": adc,
            # Legacy keys (backward compat)
            "reported_input_ms": input_latency_ms,
            "onset_detector_ms": onset_delay_ms,
        }

        # Persist to config for overlay/calibration screens that read config
        # directly without a running stream.
        try:
            self.config.latency_breakdown = LatencyBreakdown(
                input_latency_ms=input_latency_ms,
                output_latency_ms=output_latency_ms,
                detector_window_ms=detector_window_ms,
                stabilizer_confirmation_ms=consensus_delay_ms,
                render_display_ms=render_display_ms,
                manual_or_loopback_trim_ms=trim_ms,
                total_latency_ms=total_latency_ms,
                adc_timestamped=adc,
            ).to_dict()
        except Exception:
            pass  # best-effort; don't break diagnostics over a persist failure

        return result

    def get_drift_report(self) -> dict[str, float | int]:
        """Return clock drift statistics for the current session."""
        return self._drift_monitor.get_drift_report()

    def stream_time_ms(self) -> float:
        """Return captured stream time in milliseconds.

        This is based on samples written by the audio callback, not samples
        accepted by the detector worker queue.
        """
        if self._ring_sample_rate <= 0:
            return 0.0
        return self._ring_write_sample / self._ring_sample_rate * 1000.0

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

    def _write_audio_ring(self, samples: np.ndarray) -> None:
        """Write a callback block into the SPSC verification ring.

        Absolute sample indices determine placement, so oversized callback
        blocks and wraparound preserve the ``sample_idx % ring_len`` mapping.
        """
        n = len(samples)
        ring_len = len(self._audio_ring)
        if n <= 0 or ring_len <= 0:
            return

        self._ring_sequence += 1  # odd: write in progress
        try:
            write_start = self._ring_write_sample
            write_end = write_start + n
            data = samples
            data_start = write_start
            if n >= ring_len:
                data = samples[-ring_len:]
                data_start = write_end - ring_len

            start_idx = data_start % ring_len
            first = min(len(data), ring_len - start_idx)
            self._audio_ring[start_idx:start_idx + first] = data[:first]
            remaining = len(data) - first
            if remaining:
                self._audio_ring[:remaining] = data[first:]

            self._ring_write_sample = write_end
            self._ring_write_idx = write_end % ring_len
        finally:
            self._ring_sequence += 1  # even: stable snapshot

    def _copy_ring_range(
        self,
        start_sample: int,
        end_sample: int,
        *,
        max_retries: int = 4,
    ) -> np.ndarray | None:
        """Copy an absolute sample range from a stable ring snapshot.

        The callback remains wait-free. If it writes while a snapshot is being
        copied, the reader discards that copy and retries a bounded number of
        times.
        """
        total_samples = end_sample - start_sample
        if total_samples <= 0:
            return None

        for _ in range(max_retries):
            sequence_before = self._ring_sequence
            if sequence_before & 1:
                continue

            ring = self._audio_ring
            ring_len = len(ring)
            write_sample = self._ring_write_sample
            if ring_len == 0:
                return None

            if (start_sample < write_sample - ring_len
                    or end_sample > write_sample):
                if sequence_before == self._ring_sequence:
                    return None
                continue

            window = np.empty(total_samples, dtype=np.float32)
            start_idx = start_sample % ring_len
            first = min(total_samples, ring_len - start_idx)
            window[:first] = ring[start_idx:start_idx + first]
            if first < total_samples:
                window[first:] = ring[:total_samples - first]

            sequence_after = self._ring_sequence
            if sequence_before == sequence_after and not (sequence_after & 1):
                return window

        return None

    def get_recent_audio(
        self,
        ms_before: float,
        ms_after: float,
        anchor_ms: float | None = None,
    ) -> np.ndarray:
        """Extract a window of raw audio from the ring buffer.

        Returns a copy centered on ``anchor_ms`` (defaults to current captured
        stream time). An empty array means the requested range is unavailable
        or a stable snapshot could not be obtained immediately.
        """
        sample_rate = self._ring_sample_rate
        if sample_rate <= 0:
            return np.zeros(0, dtype=np.float32)
        if anchor_ms is None:
            anchor_ms = self.stream_time_ms()

        before_samples = int(sample_rate * ms_before / 1000.0)
        after_samples = int(sample_rate * ms_after / 1000.0)
        anchor_sample = int(sample_rate * anchor_ms / 1000.0)
        window = self._copy_ring_range(
            anchor_sample - before_samples,
            anchor_sample + after_samples,
        )
        if window is None:
            return np.zeros(0, dtype=np.float32)
        return window

    def get_window_between(
        self,
        start_ms: float,
        end_ms: float,
    ) -> np.ndarray | None:
        """Extract a precise ``[start_ms, end_ms)`` capture-stream window.

        Returns ``None`` until the complete requested range is available.
        """
        sample_rate = self._ring_sample_rate
        if sample_rate <= 0:
            return None
        start_sample = int(sample_rate * start_ms / 1000.0)
        end_sample = int(sample_rate * end_ms / 1000.0)
        return self._copy_ring_range(start_sample, end_sample)


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

        Subsequent worker blocks append mono chunks until
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

