"""Tests for pickhero.audio.note_utils."""

import pytest
from pickhero.audio.note_utils import (
    freq_to_midi,
    midi_to_freq,
    midi_to_name,
    name_to_midi,
    fret_to_midi,
    midi_to_fret_options,
    semitone_distance,
    is_in_guitar_range,
    STANDARD_TUNING,
)


class TestFreqToMidi:
    def test_a4(self):
        assert freq_to_midi(440.0) == 69

    def test_middle_c(self):
        # C4 = MIDI 60, ~261.63 Hz
        assert freq_to_midi(261.63) == 60

    def test_low_e_guitar(self):
        # E2 = MIDI 40, ~82.41 Hz
        assert freq_to_midi(82.41) == 40

    def test_high_e_guitar(self):
        # E4 = MIDI 64, ~329.63 Hz
        assert freq_to_midi(329.63) == 64

    def test_invalid_zero(self):
        assert freq_to_midi(0) == -1

    def test_invalid_negative(self):
        assert freq_to_midi(-100) == -1

    def test_a3(self):
        # A3 = MIDI 57, 220 Hz
        assert freq_to_midi(220.0) == 57

    def test_a5(self):
        # A5 = MIDI 81, 880 Hz
        assert freq_to_midi(880.0) == 81


class TestMidiToFreq:
    def test_a4(self):
        assert midi_to_freq(69) == pytest.approx(440.0)

    def test_a3(self):
        assert midi_to_freq(57) == pytest.approx(220.0)

    def test_middle_c(self):
        assert midi_to_freq(60) == pytest.approx(261.626, rel=1e-3)


class TestMidiToName:
    def test_a4(self):
        assert midi_to_name(69) == "A4"

    def test_middle_c(self):
        assert midi_to_name(60) == "C4"

    def test_low_e(self):
        assert midi_to_name(40) == "E2"

    def test_high_e(self):
        assert midi_to_name(64) == "E4"

    def test_sharp(self):
        assert midi_to_name(61) == "C#4"

    def test_invalid(self):
        assert midi_to_name(-1) == "?"
        assert midi_to_name(128) == "?"

    def test_open_strings_standard(self):
        """Verify all 6 open string notes in standard tuning."""
        assert midi_to_name(40) == "E2"   # string 6
        assert midi_to_name(45) == "A2"   # string 5
        assert midi_to_name(50) == "D3"   # string 4
        assert midi_to_name(55) == "G3"   # string 3
        assert midi_to_name(59) == "B3"   # string 2
        assert midi_to_name(64) == "E4"   # string 1


class TestNameToMidi:
    def test_a4(self):
        assert name_to_midi("A4") == 69

    def test_middle_c(self):
        assert name_to_midi("C4") == 60

    def test_sharp(self):
        assert name_to_midi("C#4") == 61

    def test_flat(self):
        assert name_to_midi("Db4") == 61

    def test_low_e(self):
        assert name_to_midi("E2") == 40

    def test_invalid(self):
        assert name_to_midi("") == -1
        assert name_to_midi("X") == -1
        assert name_to_midi("Z9") == -1

    def test_roundtrip(self):
        """midi_to_name -> name_to_midi should round-trip for any valid note."""
        for midi in range(0, 128):
            name = midi_to_name(midi)
            if "#" not in name:  # flats won't roundtrip through name_to_midi
                assert name_to_midi(name) == midi


class TestFretToMidi:
    def test_open_strings(self):
        for string, expected_midi in STANDARD_TUNING.items():
            assert fret_to_midi(string, 0) == expected_midi

    def test_fifth_fret_low_e(self):
        # 5th fret on low E = A2 = MIDI 45
        assert fret_to_midi(6, 5) == 45

    def test_twelfth_fret_high_e(self):
        # 12th fret on high E = E5 = MIDI 76
        assert fret_to_midi(1, 12) == 76

    def test_invalid_string(self):
        assert fret_to_midi(13, 0) == -1
        assert fret_to_midi(0, 0) == -1

    def test_seventh_string_drop_a(self):
        # Drop-A 7-string: low A1 = MIDI 33
        assert fret_to_midi(7, 0, {7: 33}) == 33
        assert fret_to_midi(7, 2, {7: 33}) == 35

    def test_custom_tuning_fallback_to_standard(self):
        # Only override string 6; others fall back to STANDARD_TUNING.
        assert fret_to_midi(6, 0, {6: 38}) == 38
        assert fret_to_midi(1, 0, {6: 38}) == 64

class TestMidiToFretOptions:
    def test_open_low_e(self):
        # E2 (40) is only available as open 6th string
        options = midi_to_fret_options(40)
        assert (6, 0) in options

    def test_a2_two_options(self):
        # A2 (45) = string 5 fret 0, or string 6 fret 5
        options = midi_to_fret_options(45)
        assert (5, 0) in options
        assert (6, 5) in options

    def test_high_note_limited_options(self):
        # E5 (76) = string 1 fret 12, string 2 fret 17
        options = midi_to_fret_options(76)
        assert (1, 12) in options
        assert (2, 17) in options

    def test_out_of_range(self):
        # Way above guitar range
        options = midi_to_fret_options(120)
        assert len(options) == 0


class TestSemitoneDistance:
    def test_same_note(self):
        assert semitone_distance(60, 60) == 0

    def test_one_semitone(self):
        assert semitone_distance(60, 61) == 1
        assert semitone_distance(61, 60) == 1

    def test_octave(self):
        assert semitone_distance(60, 72) == 12


class TestGuitarRange:
    def test_in_range(self):
        assert is_in_guitar_range(23) is True   # low B0 (5-string bass)
        assert is_in_guitar_range(40) is True   # low E
        assert is_in_guitar_range(64) is True   # high E open
        assert is_in_guitar_range(88) is True   # high E 24th fret
        assert is_in_guitar_range(96) is True   # generous headroom

    def test_out_of_range(self):
        assert is_in_guitar_range(22) is False
        assert is_in_guitar_range(97) is False
        assert is_in_guitar_range(0) is False
