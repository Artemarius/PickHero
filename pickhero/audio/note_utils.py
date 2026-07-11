"""Note conversion utilities.

Frequency <-> MIDI note number <-> note name, plus guitar string/fret mapping.
All internal note representation uses MIDI note numbers (0-127).
"""

import math

# A4 reference frequency
A4_FREQ = 440.0
A4_MIDI = 69

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard guitar tuning: string 1 (high E) to string 6 (low E)
# pyguitarpro uses 1-indexed strings: 1=high E, 6=low E
STANDARD_TUNING = {
    1: 64,  # E4
    2: 59,  # B3
    3: 55,  # G3
    4: 50,  # D3
    5: 45,  # A2
    6: 40,  # E2
}
# Guitar/bass range: low B0 of a 5-string bass (23) to high E 24th fret on guitar (88).
# Covers 4-8 string guitars and 4-6 string basses with generous headroom.
GUITAR_MIDI_MIN = 23
GUITAR_MIDI_MAX = 96

# Maximum string count we will render / validate.
MAX_STRINGS = 12

# Number of frets on a standard guitar
MAX_FRETS = 24


def freq_to_midi(freq: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number.

    Uses: round(12 * log2(freq / 440) + 69)
    Returns -1 for invalid frequencies (<= 0).
    """
    if freq <= 0:
        return -1
    return round(12 * math.log2(freq / A4_FREQ) + A4_MIDI)


def midi_to_freq(midi_note: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return A4_FREQ * (2 ** ((midi_note - A4_MIDI) / 12))


def midi_to_name(midi_note: int) -> str:
    """Convert MIDI note number to note name with octave (e.g. 'E2', 'C#4')."""
    if midi_note < 0 or midi_note > 127:
        return "?"
    note = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note}{octave}"


def name_to_midi(name: str) -> int:
    """Convert note name with octave to MIDI number (e.g. 'E2' -> 40, 'C#4' -> 61).

    Returns -1 if the name can't be parsed.
    """
    name = name.strip()
    if len(name) < 2:
        return -1

    # Extract note and octave
    if len(name) >= 3 and name[1] in ("#", "b"):
        note_part = name[:2]
        octave_part = name[2:]
    else:
        note_part = name[0]
        octave_part = name[1:]

    # Handle flats by converting to sharps
    flat_to_sharp = {
        "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
        "Ab": "G#", "Bb": "A#", "Cb": "B",
    }
    if note_part in flat_to_sharp:
        note_part = flat_to_sharp[note_part]

    if note_part not in NOTE_NAMES:
        return -1

    try:
        octave = int(octave_part)
    except ValueError:
        return -1

    return NOTE_NAMES.index(note_part) + (octave + 1) * 12


def fret_to_midi(string: int, fret: int, tuning: dict[int, int] | None = None) -> int:
    """Convert guitar string + fret to MIDI note number.

    Args:
        string: String number (1-N, where 1 = highest pitched string).
        fret: Fret number (0=open).
        tuning: Optional custom tuning dict {string_num: midi_note}. Missing
                strings fall back to STANDARD_TUNING when available.

    Returns:
        MIDI note number, or -1 if the string is not defined.
    """
    if string < 1 or string > MAX_STRINGS:
        return -1
    if fret < 0:
        return -1
    open_note = tuning.get(string) if tuning else None
    if open_note is None:
        open_note = STANDARD_TUNING.get(string, -1)
    if open_note < 0:
        return -1
    return open_note + fret


def midi_to_fret_options(midi_note: int, tuning: dict[int, int] | None = None) -> list[tuple[int, int]]:
    """Find all (string, fret) combinations that produce a given MIDI note.

    Accepts custom tunings for extended-range instruments.
    """
    if midi_note < 0 or midi_note > 127:
        return []
    options: list[tuple[int, int]] = []
    source = tuning if tuning else STANDARD_TUNING
    for string, open_note in source.items():
        if not 1 <= string <= MAX_STRINGS:
            continue
        fret = midi_note - open_note
        if 0 <= fret <= MAX_FRETS:
            options.append((string, fret))
    return sorted(options, key=lambda x: x[0])


def freq_to_cents_deviation(freq: float) -> tuple[int, float]:
    """Return (nearest_midi_note, cents_off) for a frequency.

    cents_off is in range [-50, +50]. Negative = flat, positive = sharp.
    Returns (-1, 0.0) for invalid frequencies.
    """
    if freq <= 0:
        return (-1, 0.0)
    midi_exact = 12 * math.log2(freq / A4_FREQ) + A4_MIDI
    midi_note = round(midi_exact)
    cents = (midi_exact - midi_note) * 100.0
    return (midi_note, cents)


def semitone_distance(midi_a: int, midi_b: int) -> int:
    """Absolute semitone distance between two MIDI notes."""
    return abs(midi_a - midi_b)


def is_in_guitar_range(midi_note: int) -> bool:
    """Check if a MIDI note falls within standard guitar range."""
    return GUITAR_MIDI_MIN <= midi_note <= GUITAR_MIDI_MAX

def cents_band(freq: float, cents: float) -> tuple[float, float]:
    """Return (lo, hi) frequency bounds for +/-cents around freq."""
    ratio = 2 ** (cents / 1200.0)
    return freq / ratio, freq * ratio
