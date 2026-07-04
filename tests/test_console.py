"""Tests for pickhero.audio.console testing harness."""

import argparse
import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from pickhero.audio.console import (
    ConsoleOptions,
    _parse_note_list,
    _run_synth_mode,
    _synthetic_signal,
    build_console_parser,
    options_from_args,
)
from pickhero.audio.detector import PitchDetector
from pickhero.config import Config


class TestNoteListParsing:
    def test_parses_comma_separated_integers(self):
        assert _parse_note_list("40,47") == [40, 47]

    def test_ignores_whitespace_and_empty_items(self):
        assert _parse_note_list(" 40 , , 47 ") == [40, 47]

    def test_returns_empty_for_empty_string(self):
        assert _parse_note_list("") == []


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
    def test_build_and_parse_defaults(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        build_console_parser(parser)
        args = parser.parse_args([])
        opts = options_from_args(args)
        assert opts.mode == "pitch"
        assert opts.target_notes == []
        assert opts.synth_duration_ms == 2000.0
        assert opts.noise_gate_db == -60.0

    def test_parse_synth_mode_with_notes(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        build_console_parser(parser)
        args = parser.parse_args([
            "--console-mode", "synth",
            "--console-notes", "40,47",
            "--console-duration", "500",
            "--console-gate", "-50",
        ])


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
        assert "MIDI [40]" in text
        assert "Chord present" in text
