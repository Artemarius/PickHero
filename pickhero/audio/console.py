"""Console-based testing harness for all audio input detectors.

Provides real-time, no-GUI verification of:
  - monophonic pitch detection (YIN)
  - onset detection
  - articulation detection
  - FFT-based chord / multi-pitch verification
  - synthetic signal injection for reproducible tests

Usage:
    python -m pickhero console                    # live pitch detection
    python -m pickhero console pitch                # explicit pitch mode
    python -m pickhero console chord E2 A2 D3       # chord verification
    python -m pickhero console synth E2 A2 D3       # synthetic signal test
    python -m pickhero console list                 # list audio input devices

Options:
    -d, --device INDEX      audio input device index
    -r, --sr SAMPLE_RATE    sample rate (default: 48000)
    -g, --gate DB           noise gate in dB (default: -60)
    --duration MS           synthetic signal duration in ms (default: 2000)
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
from pickhero.audio.note_utils import midi_to_freq, midi_to_name, name_to_midi
from pickhero.config import Config


@dataclass
class ConsoleOptions:
    mode: str  # "pitch", "chord", "synth", "list"
    device_index: int | None
    sample_rate: int
    target_notes: list[int]
    synth_duration_ms: float
    noise_gate_db: float


def _parse_notes(tokens: list[str]) -> list[int]:
    """Parse note tokens into MIDI note numbers.

    Tokens may be plain MIDI numbers ("40"), scientific note names ("E2"),
    or comma-separated groups ("E2,B2"). Duplicates are preserved.
    """
    notes: list[int] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                notes.append(int(part))
            except ValueError:
                midi = name_to_midi(part)
                if midi < 0:
                    raise ValueError(f"Invalid note: {part!r}")
                notes.append(midi)
    return notes


def _format_notes(notes: list[int]) -> str:
    """Return a human-readable list of notes, e.g. 'E2 B2 (40, 47)'."""
    names = " ".join(midi_to_name(n) for n in notes)
    numbers = ", ".join(str(n) for n in notes)
    return f"{names} ({numbers})"


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


def _print_device_list() -> None:
    """Print available audio input devices and mark the default."""
    devices = list_audio_devices()
    print("Available audio input devices:")
    print("-" * 60)
    for dev in devices:
        marker = " *" if dev["index"] == sd.default.device[0] else ""
        print(
            f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch, "
            f"{dev['sample_rate']:.0f}Hz, {dev['hostapi']}){marker}"
        )
    print()


def _resolve_sample_rate(device_index: int | None, fallback: int) -> int:
    """Return the device's default sample rate if available."""
    if device_index is None:
        return fallback
    try:
        info = sd.query_devices(device_index)
        rate = info.get("default_samplerate")
        if rate:
            return int(rate)
    except Exception:
        pass
    return fallback


def _run_list_mode(_opts: ConsoleOptions) -> None:
    """List audio input devices and exit."""
    _print_device_list()


def _run_pitch_mode(opts: ConsoleOptions) -> None:
    """Run the classic real-time pitch/onset/articulation console."""
    config = Config()
    config.audio.noise_gate_db = opts.noise_gate_db
    config.audio.sample_rate = _resolve_sample_rate(opts.device_index, opts.sample_rate)
    if opts.device_index is not None:
        config.audio.device_index = opts.device_index

    device_label = "default device"
    if config.audio.device_index is not None:
        try:
            info = sd.query_devices(config.audio.device_index)
            device_label = f"device {config.audio.device_index} ({info.get('name', 'unknown')})"
        except Exception:
            device_label = f"device {config.audio.device_index}"

    print(f"Listening on {device_label} — play some notes! (Ctrl+C to stop)")
    print(
        f"  Config: {config.audio.sample_rate}Hz, buf={config.audio.buf_size}, "
        f"hop={config.audio.hop_size}, confidence>={config.audio.confidence_threshold}, "
        f"noise_gate={config.audio.noise_gate_db}dB"
    )
    print("-" * 70)
    print(f"{'Time':>8}  {'Note':>5}  {'MIDI':>4}  {'Freq':>8}  {'Conf':>5}  {'Onset'}")
    print("-" * 70)

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
                    print(
                        f"{tn.timestamp_ms:7.0f}ms  {n.name:>5}{art}  "
                        f"{n.midi_note:>4}  {n.frequency:7.1f}Hz  "
                        f"{n.confidence:.2f}  {onset_marker}"
                    )
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
        print("Usage: pickhero console chord <notes>")
        print("  e.g. pickhero console chord E2 A2 D3")
        sys.exit(1)

    detector = ChordDetector(sample_rate=opts.sample_rate)

    names = [midi_to_name(n) for n in opts.target_notes]
    print(f"Chord verification target: {_format_notes(opts.target_notes)}")
    print("Play the chord — Y = present, N = missing. (Ctrl+C to stop)")
    print("  ".join(f"{name:>6}" for name in names))

    frames_since_check = 0

    def callback(indata, frames, _time_info, status):
        nonlocal frames_since_check
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        mono = indata[:, 0].copy().astype(np.float32)
        detector.push_audio(mono)
        frames_since_check += frames
        # Update display roughly every 50 ms once the buffer is primed.
        if frames_since_check >= opts.sample_rate // 20:
            frames_since_check = 0
            present = detector.verify_chord(opts.target_notes)
            line = "  ".join(
                f"{'yes':>6}" if p else f"{'no':>6}" for p in present
            )
            print(f"\r{line}", end="", flush=True)

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
        print()  # Move to a new line after the last \r output.


def _run_synth_mode(opts: ConsoleOptions) -> None:
    """Run detection on a synthetic signal (no audio device needed)."""
    if not opts.target_notes:
        print("Usage: pickhero console synth <notes>")
        print("  e.g. pickhero console synth E2 A2 D3")
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

    hop = config.audio.hop_size
    detected_notes: set[int] = set()
    detected_onsets: list[float] = []
    for i in range(0, len(signal) - hop + 1, hop):
        chunk = signal[i:i + hop]
        result = detector.process(chunk)
        if result is not None:
            detected_notes.add(result.midi_note)
            if result.is_onset:
                detected_onsets.append(i / opts.sample_rate * 1000.0)

    print(f"Synthetic signal: {_format_notes(opts.target_notes)}")
    print(f"Duration: {opts.synth_duration_ms:.1f}ms @ {opts.sample_rate}Hz")
    print()

    if detected_notes:
        print(f"Detected pitches: {_format_notes(sorted(detected_notes))}")
        print(f"Onsets: {len(detected_onsets)}")
    else:
        print("No pitches detected.")

    print()
    print("Chord verification:")
    for name, note, present in zip(
        (midi_to_name(n) for n in opts.target_notes),
        opts.target_notes,
        chord_present,
    ):
        status = "yes" if present else "no"
        print(f"  {name:>3} ({note:>2}): {status}")


# ---------------------------------------------------------------------------
# Argument parsing helpers used by pickhero.main
# ---------------------------------------------------------------------------

def build_console_parser(parent: argparse.ArgumentParser) -> None:
    """Attach console-mode arguments to the parent parser.

    The parent should be a subparser dedicated to the ``console`` command.
    """
    parent.add_argument(
        "mode",
        nargs="?",
        choices=["pitch", "chord", "synth", "list"],
        default="pitch",
        help="Console mode (default: pitch).",
    )
    parent.add_argument(
        "notes",
        nargs="*",
        help="Notes as MIDI numbers or names (e.g. E2 B2 40 47).",
    )
    parent.add_argument(
        "-d",
        "--device",
        type=int,
        default=None,
        help="Audio input device index for live modes.",
    )
    parent.add_argument(
        "-r",
        "--sr",
        type=int,
        default=48000,
        help="Sample rate (default: 48000).",
    )
    parent.add_argument(
        "-g",
        "--gate",
        type=float,
        default=-60.0,
        help="Noise gate in dB (default: -60).",
    )
    parent.add_argument(
        "--duration",
        type=float,
        default=2000.0,
        help="Synthetic signal duration in ms (default: 2000).",
    )


def options_from_args(args: argparse.Namespace) -> ConsoleOptions:
    return ConsoleOptions(
        mode=args.mode,
        device_index=args.device,
        sample_rate=args.sr,
        target_notes=_parse_notes(args.notes),
        synth_duration_ms=args.duration,
        noise_gate_db=args.gate,
    )


def run_console_test(opts: ConsoleOptions) -> None:
    """Dispatch to the requested console testing mode."""
    if opts.mode == "list":
        _run_list_mode(opts)
    elif opts.mode == "pitch":
        _run_pitch_mode(opts)
    elif opts.mode == "chord":
        _run_chord_mode(opts)
    elif opts.mode == "synth":
        _run_synth_mode(opts)
    else:
        raise ValueError(f"Unknown console mode: {opts.mode}")
