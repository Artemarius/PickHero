"""Tests for pickhero.audio.console testing harness."""

import argparse
import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from pickhero.audio.console import (
    ConsoleOptions,
    _parse_notes,
    _run_synth_mode,
    _synthetic_signal,
    build_console_parser,
    options_from_args,
)
from pickhero.audio.detector import PitchDetector
from pickhero.config import Config


class TestNoteParsing:
    def test_parses_midi_numbers(self):
        assert _parse_notes(["40", "47"]) == [40, 47]

    def test_parses_note_names(self):
        assert _parse_notes(["E2", "B2"]) == [40, 47]

    def test_parses_sharp_names(self):
        assert _parse_notes(["C#4"]) == [61]

    def test_parses_flat_names(self):
        assert _parse_notes(["Db4"]) == [61]

    def test_parses_mixed_names_and_numbers(self):
        assert _parse_notes(["E2", "47", "D3"]) == [40, 47, 50]

    def test_parses_comma_separated_tokens(self):
        assert _parse_notes(["E2,B2", "D3"]) == [40, 47, 50]

    def test_ignores_whitespace(self):
        assert _parse_notes(["  E2 ", " 47 "]) == [40, 47]

    def test_returns_empty_for_empty_list(self):
        assert _parse_notes([]) == []

    def test_raises_on_invalid_note(self):
        with pytest.raises(ValueError, match="Invalid note"):
            _parse_notes(["X9"])


class TestSyntheticSignal:
    def test_signal_contains_target_frequencies(self):
        sr = 48000
        duration_ms = 1000.0
        signal = _synthetic_signal([40], sr, duration_ms)

        assert len(signal) == int(sr * duration_ms / 1000.0)
        assert signal.dtype == np.float32
        assert np.max(np.abs(signal)) <= 1.0

        # FFT should show a peak near E2 (~82.4 Hz).
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
        peak_idx = np.argmax(np.abs(fft))
        assert 75.0 <= freqs[peak_idx] <= 90.0

    def test_pitch_detector_detects_synthesized_note(self):
        signal = _synthetic_signal([64], 48000, 1000.0)
        config = Config()
        detector = PitchDetector(
            sample_rate=48000,
            buf_size=config.audio.buf_size,
            hop_size=config.audio.hop_size,
            confidence_threshold=0.5,
            noise_gate_db=-80.0,
        )

        hop = config.audio.hop_size
        detected = []
        for i in range(0, len(signal) - hop + 1, hop):
            result = detector.process(signal[i : i + hop])
            if result is not None:
                detected.append(result.midi_note)

        # The detector should report the synthesized note (MIDI 64) at some point.
        assert 64 in detected or any(abs(m - 64) <= 1 for m in detected)


class TestArgumentParsing:
    def _parse(self, *argv: str) -> ConsoleOptions:
        parser = argparse.ArgumentParser(allow_abbrev=False)
        build_console_parser(parser)
        return options_from_args(parser.parse_args(list(argv)))

    def test_defaults_to_pitch_mode(self):
        opts = self._parse()
        assert opts.mode == "pitch"
        assert opts.target_notes == []
        assert opts.synth_duration_ms == 2000.0
        assert opts.noise_gate_db == -60.0
        assert opts.sample_rate == 48000

    def test_explicit_mode_with_note_names(self):
        opts = self._parse("chord", "E2", "A2", "D3")
        assert opts.mode == "chord"
        assert opts.target_notes == [40, 45, 50]

    def test_synth_mode_with_notes_and_options(self):
        opts = self._parse("synth", "E2,B2", "--duration", "500", "--gate", "-50")
        assert opts.mode == "synth"
        assert opts.target_notes == [40, 47]
        assert opts.synth_duration_ms == 500.0
        assert opts.noise_gate_db == -50.0

    def test_short_flags(self):
        opts = self._parse("pitch", "-d", "2", "-r", "44100", "-g", "-45")
        assert opts.device_index == 2
        assert opts.sample_rate == 44100
        assert opts.noise_gate_db == -45.0

    def test_mixed_notes_and_numbers(self):
        opts = self._parse("synth", "E2", "47", "D3")
        assert opts.target_notes == [40, 47, 50]


class TestSynthModeOutput:
    def test_synth_mode_prints_expected_notes(self):
        opts = ConsoleOptions(
            mode="synth",
            device_index=None,
            sample_rate=48000,
            target_notes=[40],
            synth_duration_ms=500.0,
            noise_gate_db=-80.0,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            _run_synth_mode(opts)

        text = out.getvalue()
        assert "Synthetic signal" in text
        assert "E2" in text
        assert "Chord verification" in text
        assert "yes" in text  # E2 should be detected as present

    def test_synth_mode_with_note_name_in_output(self):
        opts = ConsoleOptions(
            mode="synth",
            device_index=None,
            sample_rate=48000,
            target_notes=[64],  # E4
            synth_duration_ms=500.0,
            noise_gate_db=-80.0,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            _run_synth_mode(opts)

        text = out.getvalue()
        assert "E4" in text


class TestListMode:
    def test_list_mode_runs_without_error(self, monkeypatch):
        """List mode should call _print_device_list without crashing."""
        from pickhero.audio import console

        called = []
        monkeypatch.setattr(console, "_print_device_list", lambda: called.append(True))

        opts = ConsoleOptions(
            mode="list",
            device_index=None,
            sample_rate=48000,
            target_notes=[],
            synth_duration_ms=2000.0,
            noise_gate_db=-60.0,
        )
        console._run_list_mode(opts)
        assert called == [True]
