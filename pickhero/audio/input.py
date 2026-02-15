"""Audio input capture using sounddevice.

Runs a sounddevice InputStream that feeds audio buffers to the pitch detector.
Detected notes are pushed to a thread-safe queue for consumption by the main thread.
"""

import queue
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

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

        self.detector = PitchDetector(
            buf_size=ac.buf_size,
            hop_size=ac.hop_size,
            sample_rate=ac.sample_rate,
            confidence_threshold=ac.confidence_threshold,
            onset_threshold=ac.onset_threshold,
            noise_gate_db=ac.noise_gate_db,
        )
        self.note_queue: queue.Queue[TimestampedNote] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._start_time: float = 0.0

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Sounddevice callback — runs in audio thread."""
        if status:
            # Overflow or other issue — skip this buffer
            return

        # indata shape: (frames, channels) — take first channel
        mono = indata[:, 0].copy()

        # Process in hop_size chunks
        hop = self.detector.hop_size
        for i in range(0, len(mono) - hop + 1, hop):
            chunk = mono[i:i + hop]
            result = self.detector.process(chunk)
            if result is not None:
                elapsed_ms = (time.perf_counter() - self._start_time) * 1000
                self.note_queue.put(TimestampedNote(note=result, timestamp_ms=elapsed_ms))

    def start(self):
        """Start audio capture."""
        ac = self.config.audio
        self.detector.reset()

        # Drain any leftover notes
        while not self.note_queue.empty():
            try:
                self.note_queue.get_nowait()
            except queue.Empty:
                break

        self._start_time = time.perf_counter()
        self._stream = sd.InputStream(
            device=ac.device_index,
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
            config.audio.device_index = int(choice)
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
