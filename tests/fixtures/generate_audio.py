"""Generate synthetic guitar-like audio fixtures for accuracy regression tests.

Uses only numpy and the stdlib `wave` module — no new dependencies. The fixtures
are deterministic: running this script twice produces byte-identical WAVs.

Synthesis model:
- Additive harmonics (fundamental + N harmonics) with decay envelope
- Noise floor (white noise at -75 dBFS)
- Optional palm-mute (short decay) and harmonic (weak fundamental) variants

Run:  python tests/fixtures/generate_audio.py
"""

from __future__ import annotations

import math
import os
import struct
import wave

import numpy as np

SAMPLE_RATE = 44100
BIT_DEPTH = 16  # 16-bit PCM


def _midi_to_freq(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _guitar_note(
    midi: int,
    duration_s: float = 0.5,
    sr: int = SAMPLE_RATE,
    harmonics: int = 5,
    decay: float = 6.0,
    noise_db: float = -75.0,
    palm_mute: bool = False,
    harmonic_fret: int | None = None,
) -> np.ndarray:
    """Synthesize a guitar-like note via additive synthesis.

    harmonics: number of harmonic partials (1 = sine only).
    decay: exponential decay rate of the envelope (higher = faster decay).
    palm_mute: if True, very short decay (muted string).
    harmonic_fret: if set, produces a natural harmonic at that fret (weak fundamental).
    """
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    freq = _midi_to_freq(midi)

    if harmonic_fret is not None:
        # Natural harmonic: fundamental is weak, the harmonic at `fret` ratio dominates.
        # Fret 12 = 2x freq (octave), fret 7 = 3x freq (octave+fifth), fret 5 = 4x freq.
        harmonic_ratio = 2.0 ** (harmonic_fret / 12.0)
        signal = np.zeros(n, dtype=np.float64)
        # Weak fundamental
        signal += 0.15 * np.sin(2 * np.pi * freq * t)
        # Strong harmonic
        signal += 0.85 * np.sin(2 * np.pi * freq * harmonic_ratio * t)
        # A few upper partials, very weak
        for k in range(2, 4):
            signal += 0.05 * np.sin(2 * np.pi * freq * harmonic_ratio * k * t) / k
    else:
        # Additive synthesis: fundamental + harmonics with decreasing amplitude
        signal = np.zeros(n, dtype=np.float64)
        for k in range(1, harmonics + 1):
            amplitude = 1.0 / k
            signal += amplitude * np.sin(2 * np.pi * freq * k * t)

    # Decay envelope (exponential)
    if palm_mute:
        env = np.exp(-decay * 4.0 * t)  # much faster decay
    else:
        env = np.exp(-decay * t)
    signal *= env

    # Attack: short rise to avoid click
    attack_n = int(0.005 * sr)
    if attack_n > 0 and attack_n < n:
        signal[:attack_n] *= np.linspace(0, 1, attack_n)

    # Noise floor
    noise_amp = 10.0 ** (noise_db / 20.0)
    noise = np.random.RandomState(42).randn(n) * noise_amp
    signal += noise

    # Normalize to -3 dBFS peak
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    return signal.astype(np.float32)


def _clicks(positions_ms: list[float], duration_ms: float = 30.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate click train at known ms positions (for timing tests).

    Uses short (30 ms) transient bursts with sharp attack/decay. Crucially,
    there is NO continuous noise floor — the signal is pure silence between
    clicks. aubio's onset detector needs the signal to drop to silence between
    onsets to fire reliably; a continuous noise floor prevents re-triggering.
    """
    total_n = int((max(positions_ms) + duration_ms + 200) / 1000.0 * sr)
    signal = np.zeros(total_n, dtype=np.float32)
    click_n = int(duration_ms / 1000.0 * sr)
    t = np.arange(click_n) / sr
    # Sharp transient: 1000 Hz burst with fast decay
    click = 0.7 * np.sin(2 * np.pi * 1000 * t) * np.exp(-80 * t)
    for pos_ms in positions_ms:
        start = int(pos_ms / 1000.0 * sr)
        end = start + click_n
        if end <= total_n:
            signal[start:end] += click
    # No noise floor — silence between clicks lets the onset detector re-trigger
    return signal


def _chord(midis: list[int], duration_s: float = 0.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize a chord by summing individual guitar notes."""
    total = np.zeros(int(duration_s * sr), dtype=np.float32)
    for m in midis:
        total += _guitar_note(m, duration_s=duration_s, sr=sr)
    peak = np.max(np.abs(total))
    if peak > 0:
        total = total / peak * 0.7
    return total


def _clicks(positions_ms: list[float], duration_ms: float = 50.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate click train at known ms positions (for timing tests)."""
    total_n = int((max(positions_ms) + duration_ms + 200) / 1000.0 * sr)
    signal = np.zeros(total_n, dtype=np.float32)
    click_n = int(duration_ms / 1000.0 * sr)
    t = np.arange(click_n) / sr
    # Sharp click: 1000 Hz burst with fast decay
    click = 0.7 * np.sin(2 * np.pi * 1000 * t) * np.exp(-50 * t)
    for pos_ms in positions_ms:
        start = int(pos_ms / 1000.0 * sr)
        end = start + click_n
        if end <= total_n:
            signal[start:end] += click
    # Noise floor
    noise = np.random.RandomState(42).randn(total_n) * (10.0 ** (-75.0 / 20.0))
    signal += noise.astype(np.float32)
    return signal


def _save_wav(path: str, signal: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Save float32 numpy array as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert to int16
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a noise-burst-only signal (no pitch) for muted strum tests."""
    n = int(duration_s * sr)
    noise = np.random.RandomState(42).randn(n) * 0.3
    env = np.exp(-15 * np.arange(n) / sr)
    return (noise * env).astype(np.float32)


def generate_all(base_dir: str | None = None) -> None:
    """Generate the full fixture corpus."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "audio")

    sr = SAMPLE_RATE

    # single_notes/
    _save_wav(os.path.join(base_dir, "single_notes", "clean_E2.wav"),
              _guitar_note(40, duration_s=0.5, sr=sr), sr)
    _save_wav(os.path.join(base_dir, "single_notes", "clean_A2.wav"),
              _guitar_note(45, duration_s=0.5, sr=sr), sr)
    _save_wav(os.path.join(base_dir, "single_notes", "palm_muted_E2.wav"),
              _guitar_note(40, duration_s=0.3, sr=sr, palm_mute=True), sr)
    # 12th fret harmonic of E2 = E4 (MIDI 64), weak fundamental
    _save_wav(os.path.join(base_dir, "single_notes", "harmonic_12th_fret.wav"),
              _guitar_note(40, duration_s=0.5, sr=sr, harmonic_fret=12), sr)

    # chords/
    _save_wav(os.path.join(base_dir, "chords", "power_chord_E5_full.wav"),
              _chord([40, 47], duration_s=0.5, sr=sr), sr)  # E2 + B2
    _save_wav(os.path.join(base_dir, "chords", "power_chord_root_only.wav"),
              _guitar_note(40, duration_s=0.5, sr=sr), sr)  # E2 only
    _save_wav(os.path.join(base_dir, "chords", "open_E_major.wav"),
              _chord([40, 47, 52], duration_s=0.5, sr=sr), sr)  # E2 + B2 + E3
    _save_wav(os.path.join(base_dir, "chords", "muted_strum.wav"),
              _silence(0.3, sr=sr), sr)

    # timing/
    _save_wav(os.path.join(base_dir, "timing", "click_aligned_onsets.wav"),
              _clicks([500, 1000, 1500, 2000], sr=sr), sr)
    _save_wav(os.path.join(base_dir, "timing", "early_25ms.wav"),
              _clicks([475, 975, 1475, 1975], sr=sr), sr)
    _save_wav(os.path.join(base_dir, "timing", "late_25ms.wav"),
              _clicks([525, 1025, 1525, 2025], sr=sr), sr)

    print(f"Generated fixture corpus in {base_dir}")


if __name__ == "__main__":
    generate_all()
