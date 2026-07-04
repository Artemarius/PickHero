"""Console-based testing harness for all audio input detectors.

Provides real-time, no-GUI verification of:
  - monophonic pitch detection (YIN)
  - onset detection
  - articulation detection
  - FFT-based chord / multi-pitch verification
  - synthetic signal injection for reproducible tests
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from pickhero.audio.chord_detector import ChordDetector
from pickhero.audio.detector import PitchDetector
from pickhero.audio.input import AudioCapture, list_audio_devices
from pickhero.audio.note_utils import midi_to_freq
from pickhero.config import Config


@dataclass
class ConsoleOptions:
    mode: str  # "pitch", "chord", "synth"
    device_index: int | None
    sample_rate: int
    target_notes: list[int]
    synth_duration_ms: float
    noise_gate_db: float


def _parse_note_list(value: str) -> list[int]:
    """Parse a comma-separated list of MIDI note numbers."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _synthetic_signal(
    midi_notes: list[int],
    sample_rate: int,
    duration_ms: float,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a synthetic guitar-like signal for the given MIDI notes."""
    samples = int(sample_rate * duration_ms / 1000.0)
    t = np.arange(samples) / sample_rate
    signal = np.zeros(samples, dtype=np.float32)

    for midi in midi_notes:
        freq = midi_to_freq(midi)
        # Sum harmonics with decaying amplitude to approximate a plucked string.
        for h in range(1, 9):
            signal += np.sin(2 * np.pi * freq * h * t) / h

    # Normalize and apply a quick attack/decay envelope.
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * amplitude

    attack = min(int(0.01 * sample_rate), samples // 4)
    if attack > 0:
        env = np.ones(samples, dtype=np.float32)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[attack:] = np.exp(-np.arange(samples - attack) / (sample_rate * 0.3))
        signal = signal * env

    return signal.astype(np.float32)


def _run_pitch_mode(opts: ConsoleOptions) -> None:
    """Run the classic real-time pitch/onset/articulation console."""
    config = Config()
    config.audio.sample_rate = opts.sample_rate
    config.audio.noise_gate_db = opts.noise_gate_db

    if opts.device_index is not None:
        config.audio.device_index = opts.device_index
        info = sd.query_devices(opts.device_index)
        if info.get("default_samplerate"):
            config.audio.sample_rate = int(info["default_samplerate"])
    else:
        devices = list_audio_devices()
        print("Available audio input devices:")
        print("-" * 60)
        for dev in devices:
            marker = " *" if dev["index"] == sd.default.device[0] else ""
            print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch, "
                  f"{dev['sample_rate']:.0f}Hz, {dev['hostapi']}){marker}")
        print()
        choice = input("Select device index (Enter for default): ").strip()
        if choice:
            try:
                idx = int(choice)
                config.audio.device_index = idx
                info = sd.query_devices(idx)
                if info.get("default_samplerate"):
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

    last_note_name = ""
    last_xrun_print = 0
    try:
        while True:
            notes = capture.get_notes()
            for tn in notes:
                n = tn.note
                note_id = f"{n.name}:{n.articulation or ''}"
                if n.is_onset or note_id != last_note_name:
                    art = f" [{n.articulation}]" if n.articulation else ""
                    onset_marker = ">>>" if n.is_onset else "   "
                    print(f"{tn.timestamp_ms:7.0f}ms  {n.name:>5}{art}  "
                          f"{n.midi_note:>4}  {n.frequency:7.1f}Hz  "
                          f"{n.confidence:.2f}  {onset_marker}")
                    last_note_name = note_id

            xruns = capture.get_xrun_count()
            if xruns != last_xrun_print:
                print(f"[audio glitches: {xruns}]")
                last_xrun_print = xruns

            time.sleep(0.01)
    finally:
        capture.stop()


def _run_chord_mode(opts: ConsoleOptions) -> None:
    """Run FFT-based chord verification for a target set of MIDI notes."""
    if not opts.target_notes:
        print("Chord mode requires target notes, e.g. --console-notes 40,47")
        sys.exit(1)

    detector = ChordDetector(sample_rate=opts.sample_rate)

    names = ",".join(str(n) for n in opts.target_notes)
    print(f"Chord verification target: {names}")
    print("Play the chord — each line shows which notes are present in the spectrum.")
    print("(Ctrl+C to stop)")

    frames_since_check = 0

    def callback(indata, frames, time_info, status):
        nonlocal frames_since_check
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        mono = indata[:, 0].copy().astype(np.float32)
        detector.push_audio(mono)
        frames_since_check += frames
        # Run verification roughly every 50 ms once the buffer is primed.
        if frames_since_check >= opts.sample_rate // 20:
            frames_since_check = 0
            present = detector.verify_chord(opts.target_notes)
            present_str = "".join("Y" if p else "N" for p in present)
            print(f"{time.time() % 1000:8.3f}s  {present_str}  {names}")

    try:
        with sd.InputStream(
            samplerate=opts.sample_rate,
            device=opts.device_index,
            channels=1,
            blocksize=1024,
            callback=callback,
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def _run_synth_mode(opts: ConsoleOptions) -> None:
    """Run detection on a synthetic signal (no audio device needed)."""
    if not opts.target_notes:
        print("Synth mode requires notes, e.g. --console-notes 40,47")
        sys.exit(1)

    signal = _synthetic_signal(
        opts.target_notes,
        opts.sample_rate,
        opts.synth_duration_ms,
    )

    config = Config()
    config.audio.sample_rate = opts.sample_rate
    config.audio.noise_gate_db = opts.noise_gate_db

    detector = PitchDetector(
        sample_rate=opts.sample_rate,
        buf_size=config.audio.buf_size,
        hop_size=config.audio.hop_size,
        confidence_threshold=config.audio.confidence_threshold,
        noise_gate_db=opts.noise_gate_db,
    )
    chord_detector = ChordDetector(sample_rate=opts.sample_rate)
    chord_detector.push_audio(signal)
    chord_present = chord_detector.verify_chord(opts.target_notes)
    chord_str = "".join("Y" if p else "N" for p in chord_present)

    hop = config.audio.hop_size
    print(f"Synthetic signal: MIDI {opts.target_notes}, {opts.synth_duration_ms}ms, "
          f"{opts.sample_rate}Hz")
    print("-" * 70)
    print(f"{'Time':>8}  {'Pitch':>6}  {'MIDI':>4}  {'Freq':>8}  {'Conf':>5}  "
          f"{'Onset'}  {'Chord present'}")
    print("-" * 70)

    for i in range(0, len(signal) - hop + 1, hop):
        chunk = signal[i:i + hop]
        result = detector.process(chunk)
        elapsed_ms = i / opts.sample_rate * 1000.0
        if result is not None:
            onset = ">>>" if result.is_onset else "   "
            print(f"{elapsed_ms:7.1f}ms  {result.name:>6}  {result.midi_note:>4}  "
                  f"{result.frequency:7.1f}Hz  {result.confidence:.2f}  {onset}  {chord_str}")


# ---------------------------------------------------------------------------
# Argument parsing helpers used by pickhero.main
# ---------------------------------------------------------------------------

def build_console_parser(parent: argparse.ArgumentParser) -> None:
    """Attach console-mode sub-arguments to the main parser."""
    group = parent.add_argument_group("console mode options")
    group.add_argument(
        "--console-mode",
        choices=["pitch", "chord", "synth"],
        default="pitch",
        help="Console testing mode (default: pitch).",
    )
    group.add_argument(
        "--console-notes",
        type=str,
        default="",
        help="Comma-separated MIDI notes for chord/synth mode, e.g. 40,47.",
    )
    group.add_argument(
        "--console-duration",
        type=float,
        default=2000.0,
        help="Synthetic signal duration in ms for synth mode (default: 2000).",
    )
    group.add_argument(
        "--console-device",
        type=int,
        default=None,
        help="Audio input device index for live modes.",
    )
    group.add_argument(
        "--console-sr",
        type=int,
        default=48000,
        help="Sample rate for console modes (default: 48000).",
    )
    group.add_argument(
        "--console-gate",
        type=float,
        default=-60.0,
        help="Noise gate in dB (default: -60).",
    )


def options_from_args(args: argparse.Namespace) -> ConsoleOptions:
    return ConsoleOptions(
        mode=args.console_mode,
        device_index=args.console_device,
        sample_rate=args.console_sr,
        target_notes=_parse_note_list(args.console_notes),
        synth_duration_ms=args.console_duration,
        noise_gate_db=args.console_gate,
    )


def run_console_test(opts: ConsoleOptions) -> None:
    """Dispatch to the requested console testing mode."""
    if opts.mode == "pitch":
        _run_pitch_mode(opts)
    elif opts.mode == "chord":
        _run_chord_mode(opts)
    elif opts.mode == "synth":
        _run_synth_mode(opts)
    else:
        raise ValueError(f"Unknown console mode: {opts.mode}")
