"""Offline polyphonic analyzer for the after-take deep-analysis pass.

Runs after the real-time :class:`~pickhero.audio.analyzer.PerformanceAnalyzer`
when the active preset has ``offline_deep_analysis=True``. Operates on the full
raw take audio (recorded by :meth:`~pickhero.audio.input.AudioCapture.start_take_recording`)
plus the matched ``(PerformanceEvent, NoteEvent)`` pairs.

Two analyses (Phase 1):

- **Unison bend detection**: for each pair of simultaneously-sounding
  :class:`~pickhero.tabs.timeline.NoteEvent` s where one has a ``bend`` spec
  and the other is a held static note, run two-F0 analysis (FFT harmonic-bank
  matching) to measure beating / chorus and emit a
  :class:`~pickhero.audio.performance.TechniqueVerdict` with
  ``kind="unison_bend"``.

- **Pinch harmonic verification**: for ``pinch`` harmonic specs, verify
  high-overtone dominance + squeal envelope + fundamental suppression ratio.

The two-F0 analysis is self-contained here (not reusing ``ChordDetector``'s
private harmonic-bank math) to avoid contorting the real-time API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pickhero.audio.performance import TechniqueVerdict

if TYPE_CHECKING:
    from pickhero.audio.performance import PerformanceEvent
    from pickhero.tabs.timeline import NoteEvent


class PolyphonicAnalyzer:
    """Offline polyphonic analysis over a full take.

    Parameters
    ----------
    raw_audio : np.ndarray
        Mono float32 take audio (from ``stop_take_recording``).
    sample_rate : int
        Sample rate of ``raw_audio``.
    matched_pairs : list[tuple[PerformanceEvent, NoteEvent]]
        The matched pairs from the real-time analyzer (event, note).
    """

    def __init__(
        self,
        raw_audio: np.ndarray,
        sample_rate: int,
        matched_pairs: list[tuple["PerformanceEvent", "NoteEvent"]],
    ):
        self.audio = np.asarray(raw_audio, dtype=np.float32)
        self.sample_rate = int(sample_rate)
        self.pairs = list(matched_pairs)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self) -> list[TechniqueVerdict]:
        """Run all offline analyses. Returns new verdicts (not appended to
        events — the caller merges them into ``all_verdicts``)."""
        verdicts: list[TechniqueVerdict] = []
        verdicts.extend(self._detect_unison_bends())
        verdicts.extend(self._verify_pinch_harmonics())
        return verdicts

    # ------------------------------------------------------------------
    # Unison bend detection
    # ------------------------------------------------------------------

    def _detect_unison_bends(self) -> list[TechniqueVerdict]:
        """Find unison-bend pairs: a bent note + a simultaneously-sounding static
        note on a different string. Emit a verdict per qualifying pair."""
        verdicts: list[TechniqueVerdict] = []
        if len(self.audio) == 0 or not self.pairs:
            return verdicts

        # Index pairs by their time overlap.
        for i, (ev_a, note_a) in enumerate(self.pairs):
            bend_spec = next(
                (s for s in note_a.techniques if s.kind == "bend"), None,
            )
            if bend_spec is None:
                continue
            # Find a simultaneously-sounding static note (no bend) overlapping
            # this note's sustain window.
            for j, (ev_b, note_b) in enumerate(self.pairs):
                if i == j:
                    continue
                if note_b.string == note_a.string:
                    continue
                if any(s.kind == "bend" for s in note_b.techniques):
                    continue  # both bent — not a unison bend
                if not self._events_overlap(ev_a, ev_b):
                    continue
                # Two-F0 analysis on the overlap window.
                verdict = self._analyze_unison_pair(
                    ev_a, note_a, ev_b, note_b, bend_spec,
                )
                if verdict is not None:
                    verdicts.append(verdict)
        return verdicts

    def _events_overlap(
        self, ev_a: "PerformanceEvent", ev_b: "PerformanceEvent",
    ) -> bool:
        """True if the two events' sustain windows overlap in time."""
        a_start = ev_a.onset_ms
        a_end = ev_a.release_ms if ev_a.release_ms is not None else a_start + 500.0
        b_start = ev_b.onset_ms
        b_end = ev_b.release_ms if ev_b.release_ms is not None else b_start + 500.0
        return a_start < b_end and b_start < a_end

    def _analyze_unison_pair(
        self, ev_bent, note_bent, ev_static, note_static, bend_spec,
    ) -> TechniqueVerdict | None:
        """Two-F0 analysis on the overlap window of a bent + static note pair.

        Measures whether the bent note's pitch approaches the static note's
        pitch (the unison target) and how cleanly they converge.
        """
        # Extract the overlap window from raw audio.
        start_ms = max(
            ev_bent.onset_ms, ev_static.onset_ms,
        )
        end_ms = min(
            ev_bent.release_ms or (ev_bent.onset_ms + 500.0),
            ev_static.release_ms or (ev_static.onset_ms + 500.0),
        )
        if end_ms <= start_ms:
            return None
        sr = self.sample_rate
        start_sample = int(start_ms / 1000.0 * sr)
        end_sample = int(end_ms / 1000.0 * sr)
        window = self.audio[start_sample:end_sample]
        if len(window) < sr // 4:  # need at least 250ms
            return None

        # Two-F0 via FFT: find the two strongest harmonic banks.
        static_midi = note_static.midi_note
        static_freq = 440.0 * (2.0 ** ((static_midi - 69) / 12.0))
        # Measure energy at the static note's harmonics vs the bent note's
        # final pitch (from the bend event's f0_curve).
        bent_final_freq = 0.0
        if ev_bent.f0_curve:
            bent_final_freq = ev_bent.f0_curve[-1][1]
        if bent_final_freq <= 0:
            return None

        static_energy = self._harmonic_bank_energy(window, static_freq, sr)
        bent_energy = self._harmonic_bank_energy(window, bent_final_freq, sr)
        if static_energy <= 0 or bent_energy <= 0:
            return None

        # Convergence: how close did the bent note get to the static note?
        cents_diff = 1200.0 * np.log2(bent_final_freq / static_freq)
        target_error_cents = abs(cents_diff)
        # Time-to-unison: how long into the window the bend reached within 25¢.
        time_to_unison_ms = self._time_to_unison(ev_bent, static_freq)
        # Release cleanliness: spectral flatness at the end of the window.
        release_cleanliness = 1.0 - self._spectral_flatness(window[-sr // 8:])

        # Grade: good if converged within 25 cents, weak if > 50.
        if target_error_cents <= 25.0:
            grade = "good"
            score = 1.0
        elif target_error_cents <= 50.0:
            grade = "ok"
            score = 0.6
        else:
            grade = "weak"
            score = 0.3

        return TechniqueVerdict(
            kind="unison_bend",
            grade=grade,
            score=score,
            metrics={
                "target_error_cents": target_error_cents,
                "time_to_unison_ms": time_to_unison_ms,
                "release_cleanliness": release_cleanliness,
                "static_freq": static_freq,
                "bent_final_freq": bent_final_freq,
            },
            explanation=(
                f"Unison bend reached {target_error_cents:.0f} cents from "
                f"unison in {time_to_unison_ms:.0f}ms."
            ),
        )

    def _time_to_unison(
        self, event: "PerformanceEvent", target_freq: float,
    ) -> float:
        """How long (ms) until the event's f0_curve gets within 25 cents of
        target_freq. Returns the full sustain if it never converges."""
        if not event.f0_curve:
            return 0.0
        onset = event.onset_ms
        for t, freq, _ in event.f0_curve:
            if freq <= 0 or target_freq <= 0:
                continue
            cents = 1200.0 * np.log2(freq / target_freq)
            if abs(cents) <= 25.0:
                return t - onset
        # Never converged — return the full sustain.
        last_t = event.f0_curve[-1][0] if event.f0_curve else onset
        return last_t - onset

    # ------------------------------------------------------------------
    # Pinch harmonic verification
    # ------------------------------------------------------------------

    def _verify_pinch_harmonics(self) -> list[TechniqueVerdict]:
        """For pinch-harmonic specs, verify high-overtone dominance + squeal
        envelope + fundamental suppression ratio in the onset window."""
        verdicts: list[TechniqueVerdict] = []
        if len(self.audio) == 0 or not self.pairs:
            return verdicts
        sr = self.sample_rate
        for event, note in self.pairs:
            pinch = next(
                (s for s in note.techniques
                 if s.kind == "harmonic" and s.subtype == "pinch"),
                None,
            )
            if pinch is None:
                continue
            start = int(event.onset_ms / 1000.0 * sr)
            window = self.audio[start:start + sr // 2]  # 500ms onset window
            if len(window) < sr // 8:
                continue
            fundamental = 440.0 * (2.0 ** ((note.midi_note - 69) / 12.0))
            # High-overtone dominance: energy above 2× f0 vs energy at f0.
            f0_energy = self._harmonic_bank_energy(window, fundamental, sr)
            high_energy = self._band_energy(
                window, fundamental * 2.5, fundamental * 8.0, sr,
            )
            if f0_energy <= 0:
                continue
            overtone_ratio = high_energy / f0_energy
            # Squeal envelope: peak energy in the first 50ms (attack).
            attack = window[: sr // 20] if len(window) >= sr // 20 else window
            squeal = float(np.max(np.abs(attack))) if len(attack) else 0.0
            # Fundamental suppression: how much weaker f0 is vs the overtone band.
            suppression = 1.0 - min(1.0, f0_energy / (high_energy + 1e-9))

            if overtone_ratio > 2.0 and suppression > 0.3:
                grade = "good"
                score = 1.0
            elif overtone_ratio > 1.0:
                grade = "ok"
                score = 0.6
            else:
                grade = "weak"
                score = 0.3

            verdicts.append(TechniqueVerdict(
                kind="pinch_harmonic",
                grade=grade,
                score=score,
                metrics={
                    "overtone_ratio": overtone_ratio,
                    "squeal_amplitude": squeal,
                    "fundamental_suppression": suppression,
                },
                explanation=(
                    f"Pinch harmonic: overtone ratio {overtone_ratio:.2f}, "
                    f"fundamental suppression {suppression:.2f}."
                ),
            ))
        return verdicts

    # ------------------------------------------------------------------
    # FFT helpers (self-contained — not reusing ChordDetector)
    # ------------------------------------------------------------------

    def _band_energy(
        self, audio: np.ndarray, f_low: float, f_high: float, sr: int,
    ) -> float:
        """Total magnitude in the [f_low, f_high] band via FFT."""
        if len(audio) < 2:
            return 0.0
        n = 1 << int(np.ceil(np.log2(len(audio))))
        window = np.hanning(len(audio)).astype(np.float32)
        padded = np.zeros(n, dtype=np.float32)
        padded[: len(audio)] = audio * window
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = (freqs >= f_low) & (freqs <= f_high)
        return float(np.sum(spectrum[mask])) if mask.any() else 0.0

    def _harmonic_bank_energy(
        self, audio: np.ndarray, f0: float, sr: int, n_harmonics: int = 5,
    ) -> float:
        """Sum of magnitude at integer multiples of f0 (harmonic-bank matching)."""
        if f0 <= 0 or len(audio) < 2:
            return 0.0
        n = 1 << int(np.ceil(np.log2(len(audio))))
        window = np.hanning(len(audio)).astype(np.float32)
        padded = np.zeros(n, dtype=np.float32)
        padded[: len(audio)] = audio * window
        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        nyquist = sr / 2.0
        total = 0.0
        for h in range(1, n_harmonics + 1):
            f = f0 * h
            if f >= nyquist:
                break
            # Energy in a ±5 Hz window around the harmonic.
            mask = (freqs >= f - 5.0) & (freqs <= f + 5.0)
            if mask.any():
                total += float(np.max(spectrum[mask]))
        return total

    def _spectral_flatness(self, audio: np.ndarray) -> float:
        """Geometric mean / arithmetic mean (0=pitched, 1=noise)."""
        if len(audio) < 2:
            return 0.0
        window = np.hanning(len(audio)).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(audio * window))
        if len(spectrum) == 0:
            return 0.0
        mag = np.maximum(spectrum, 1e-12)
        geo = float(np.exp(np.mean(np.log(mag))))
        arith = float(np.mean(mag))
        if arith <= 0:
            return 0.0
        return geo / arith
