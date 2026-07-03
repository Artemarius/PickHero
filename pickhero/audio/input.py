"""Audio input capture using sounddevice.

Runs a sounddevice InputStream that feeds audio buffers to the pitch detector.
Detected notes are pushed to a thread-safe queue for consumption by the main thread.
"""

import queue
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from pickhero.audio.chord_detector import ChordDetector
from pickhero.audio.detector import PitchDetector, DetectedNote
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
        self._stream: sd.InputStream | None = None
        self._start_time: float = 0.0
        self._signal_db: float = -120.0
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self.chord_detector = ChordDetector(sample_rate=ac.sample_rate)
        # Absolute sample offset consumed by the detector (advances by hop per process() call)
        self._detector_sample_offset: int = 0
        # Whether time_info.inputBufferAdcTime is available on this backend
        self._adc_time_available: bool | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Sounddevice callback — runs in audio thread."""
        if status:
            # Overflow or other issue — skip this buffer
            return

        # indata shape: (frames, channels) — take first channel
        mono = indata[:, 0].copy()

        # Feed chord detector (FFT-based, runs on full buffer)
        self.chord_detector.push_audio(mono)

        # Determine the ADC timestamp of this buffer's first sample.
        # On the first callback, initialize _start_time and probe backend support.
        adc_time = getattr(time_info, "inputBufferAdcTime", 0.0) if time_info else 0.0
        if self._adc_time_available is None:
            self._adc_time_available = adc_time > 0.0
            if self._adc_time_available:
                self._start_time = adc_time
            else:
                # Backend doesn't populate ADC time — fall back to wall clock
                self._start_time = time.perf_counter()
        elif self._adc_time_available and self._start_time == 0.0:
            self._start_time = adc_time

        sample_rate = self.detector.sample_rate

        # Process in hop_size chunks
        hop = self.detector.hop_size
        for i in range(0, len(mono) - hop + 1, hop):
            chunk = mono[i:i + hop]
            result = self.detector.process(chunk)
            self._signal_db = self.detector.last_signal_db
            self._tuner_freq = self.detector.last_freq
            self._tuner_confidence = self.detector.last_confidence
            if result is not None:
                elapsed_ms = self._compute_timestamp_ms(
                    result, i, adc_time, sample_rate, hop
                )
                self.note_queue.put(TimestampedNote(note=result, timestamp_ms=elapsed_ms))
            # Advance the absolute sample counter by one hop
            self._detector_sample_offset += hop

    def _compute_timestamp_ms(
        self,
        result: DetectedNote,
        chunk_offset_in_buffer: int,
        adc_time: float,
        sample_rate: int,
        hop: int,
    ) -> float:
        """Compute a low-jitter timestamp (ms from session start) for a detected note.

        Uses the audio callback's ADC time as the anchor, plus the onset's sample
        position within the stream, minus the onset detector's algorithmic delay.
        Falls back to wall-clock time if the backend doesn't provide ADC timestamps.
        """
        if not self._adc_time_available:
            # Backend without ADC time — use wall clock (legacy behavior)
            return (time.perf_counter() - self._start_time) * 1000.0

        # onset_sample is the absolute sample position since detector creation.
        # The chunk being processed started at self._detector_sample_offset - hop
        # samples into the stream. The onset's offset within this chunk is:
        onset_sample = result.onset_sample
        chunk_start_sample = self._detector_sample_offset - hop
        if onset_sample is not None and onset_sample >= chunk_start_sample:
            samples_into_stream = float(onset_sample)
        else:
            # Onset sample not available or stale — use chunk start
            samples_into_stream = float(chunk_start_sample)

        # Algorithmic delay of the onset detector (samples)
        delay_samples = float(self.detector._onset.get_delay())

        # Wall-clock time of the onset = buffer ADC time + onset offset in stream
        #   minus algorithmic delay. _start_time is the ADC time of the very first
        #   buffer, so subtracting it gives ms-from-session-start.
        onset_wall_s = (
            adc_time
            + (samples_into_stream / sample_rate)
            + (0.0 - (delay_samples / sample_rate))
        )
        return (onset_wall_s - self._start_time) * 1000.0

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

        # Resolve device name → index, updating sample_rate to match
        resolved = self._resolve_device()

        # Recreate detector with the resolved sample rate (may have changed)
        from pickhero.audio.detector import PitchDetector
        calibration = getattr(self.config, 'calibration', None) or None
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
        # _start_time will be set on first callback from time_info.inputBufferAdcTime
        self._start_time = 0.0
        self._stream = sd.InputStream(
            device=resolved,
            channels=1,
            samplerate=ac.sample_rate,
            blocksize=ac.hop_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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

    def get_notes(self) -> list[TimestampedNote]:
        """Drain all pending detected notes from the queue (non-blocking)."""
        notes = []
        while True:
            try:
                notes.append(self.note_queue.get_nowait())
            except queue.Empty:
                break
        return notes


def list_audio_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
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


def run_console_demo():
    """Interactive console demo for testing pitch detection.

    Lists audio devices, lets user pick one, then prints detected notes in real-time.
    """
    print("Available audio input devices:")
    print("-" * 50)
    devices = list_audio_devices()
    if not devices:
        print("No audio input devices found!")
        return

    for dev in devices:
        marker = " *" if dev["index"] == sd.default.device[0] else ""
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch, {dev['sample_rate']:.0f}Hz){marker}")
    print()

    choice = input("Select device index (Enter for default): ").strip()
    config = Config()
    if choice:
        try:
            idx = int(choice)
            config.audio.device_index = idx
            # Use the device's native sample rate so it works even for
            # non-standard devices (headsets, USB mics at 48k, etc.)
            info = sd.query_devices(idx)
            if info["default_samplerate"]:
                config.audio.sample_rate = int(info["default_samplerate"])
        except ValueError:
            print("Invalid input, using default device.")

    print()
    print("Listening... play some notes! (Ctrl+C to stop)")
    print(f"  Config: buf={config.audio.buf_size}, hop={config.audio.hop_size}, "
          f"confidence>={config.audio.confidence_threshold}, noise_gate={config.audio.noise_gate_db}dB")
    print("-" * 60)
    print(f"{'Time':>8}  {'Note':>5}  {'MIDI':>4}  {'Freq':>8}  {'Conf':>5}  {'Onset'}")
    print("-" * 60)

    capture = AudioCapture(config)
    capture.start()

    last_note = ""
    try:
        while True:
            notes = capture.get_notes()
            for tn in notes:
                n = tn.note
                # Only print on onset or note change to reduce spam
                current = n.name
                if n.is_onset or current != last_note:
                    onset_marker = ">>>" if n.is_onset else "   "
                    print(f"{tn.timestamp_ms:7.0f}ms  {n.name:>5}  {n.midi_note:>4}  "
                          f"{n.frequency:7.1f}Hz  {n.confidence:.2f}  {onset_marker}")
                    last_note = current
            time.sleep(0.01)  # ~100Hz polling, avoid busy-wait
    finally:
        capture.stop()
