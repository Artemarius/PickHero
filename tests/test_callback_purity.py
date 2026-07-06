"""Tests for pure ring-buffer callback + unified worker thread.

Verifies Tasks 1.2 and 1.3: the audio callback must be real-time safe
(copy only into a queue), and a unified worker thread must handle all DSP
for both portable and high_accuracy profiles.
"""

import queue
import time

import numpy as np
import pytest

aubio = pytest.importorskip("aubio")

from pickhero.audio.input import AudioCapture  # noqa: E402
from pickhero.config import Config  # noqa: E402


class _MockTimeInfo:
    """Minimal stand-in for PortAudio's time_info struct."""

    def __init__(self, adc_time: float):
        self.inputBufferAdcTime = adc_time
        self.outputBufferDacTime = 0.0
        self.currentTime = adc_time


def _make_capture(config: Config):
    """Helper: create AudioCapture, return (capture, call_cb)."""
    capture = AudioCapture(config)
    time_idx = [0]

    def call_cb(signal: np.ndarray, adc_time: float = 0.0):
        indata = signal.reshape(-1, 1).astype(np.float32)
        capture._audio_callback(indata, len(signal), _MockTimeInfo(adc_time), 0)

    return capture, call_cb


class TestCallbackPurity:
    """Task 1.2: _audio_callback must do NO DSP."""

    def test_callback_does_not_call_detector_process(self):
        """Patching detector.process to raise should NOT cause _audio_callback
        to raise — the callback must not call it at all.
        """
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        capture = AudioCapture(config)
        assert capture._engine is None  # portable mode

        original_process = capture.detector.process
        capture.detector.process = lambda x: (_ for _ in ()).throw(
            RuntimeError("detector.process must not be called from callback")
        )

        sr = 44100
        hop = 512
        burst = (
            0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 0.2) / sr)
        ).astype(np.float32)

        for i in range(0, len(burst) - hop, hop):
            chunk = burst[i:i + hop]
            indata = chunk.reshape(-1, 1).astype(np.float32)
            capture._audio_callback(
                indata, len(chunk), _MockTimeInfo(i / sr), 0
            )

        # Callback finished without raising — detector.process was not called.
        capture.detector.process = original_process

    def test_callback_submits_to_worker_queue(self):
        """Calling _audio_callback with audio must push chunks into
        _worker_in_queue (the consumer side is exercised by
        test_portable_worker_emits_stable_events).
        """
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512
        config.audio.sample_rate = 44100

        capture = AudioCapture(config)

        sr = 44100
        hop = 512
        burst = (
            0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 0.1) / sr)
        ).astype(np.float32)

        num_chunks_expected = len(burst) // hop
        assert capture._worker_in_queue.empty()

        indata = burst.reshape(-1, 1).astype(np.float32)
        capture._audio_callback(
            indata, len(burst), _MockTimeInfo(0.0), 0
        )

        # Worker queue should have received the chunks.
        count = 0
        while True:
            try:
                capture._worker_in_queue.get_nowait()
                count += 1
            except queue.Empty:
                break

        # Allow a few less due to the range(len - hop + 1) boundary.
        assert count >= num_chunks_expected - 1, (
            f"Expected ~{num_chunks_expected} chunks, got {count}"
        )

    def test_callback_no_fft_in_callback(self):
        """The callback must never call spectral_flatness (or any FFT-heavy
        code) because it runs in the real-time audio thread.
        """
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        capture = AudioCapture(config)

        fft_called = [False]
        original_flatness = capture.detector._spectral_flatness
        capture.detector._spectral_flatness = lambda x, y: (
            fft_called.__setitem__(0, True) or original_flatness(x, y)
        )

        sr = 44100
        hop = 512
        burst = (
            0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 0.05) / sr)
        ).astype(np.float32)

        indata = burst.reshape(-1, 1).astype(np.float32)
        capture._audio_callback(
            indata, len(burst), _MockTimeInfo(0.0), 0
        )

        assert not fft_called[0], (
            "_spectral_flatness was called from the callback — "
            "this belongs on the worker thread."
        )

    def test_callback_preserves_xrun_counting(self):
        """Input overflow status should increment _xrun_count."""
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        capture = AudioCapture(config)
        indata = np.zeros((512, 1), dtype=np.float32)

        capture._audio_callback(
            indata, 512, _MockTimeInfo(0.0), "Input overflow"
        )
        assert capture.get_xrun_count() >= 1

    def test_callback_preserves_take_recording(self):
        """When take recording is armed, the callback must append mono copies."""
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        capture = AudioCapture(config)
        capture.start_take_recording()

        sr = 44100
        hop = 512
        burst = (
            0.5 * np.sin(2 * np.pi * 440 * np.arange(hop) / sr)
        ).astype(np.float32)
        indata = burst.reshape(-1, 1).astype(np.float32)

        capture._audio_callback(
            indata, hop, _MockTimeInfo(0.0), 0
        )

        assert capture._take_audio is not None
        assert len(capture._take_audio) >= 1


class TestUnifiedWorker:
    """Task 1.3: Unified worker thread processes chunks and emits notes."""

    def test_worker_thread_starts_on_start(self):
        """The worker thread should be alive after start()."""
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        capture = AudioCapture(config)
        capture._start_unified_worker()
        try:
            assert capture._worker_running
            assert capture._worker_thread is not None
            assert capture._worker_thread.is_alive()
        finally:
            capture.stop()

    def test_worker_thread_stops_on_stop(self):
        """After stop(), the worker thread should no longer be running."""
        config = Config()
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512

        capture = AudioCapture(config)
        capture._start_unified_worker()
        try:
            assert capture._worker_thread.is_alive()
        finally:
            capture.stop()

        # Worker should have been joined and stopped (thread may be None).
        assert capture._worker_thread is None or not capture._worker_thread.is_alive()

    def test_portable_worker_emits_stable_events(self):
        """Feed 440 Hz sine through the worker thread and expect notes
        from get_notes() after a short wait.
        """
        config = Config()
        config.audio.profile = "portable"
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0
        config.audio.buf_size = 2048
        config.audio.hop_size = 512
        config.audio.sample_rate = 44100

        capture = AudioCapture(config)
        capture._start_unified_worker()
        try:
            sr = 44100
            hop = 512
            # Generate ~1s of 440 Hz sine wave
            tone_len = sr
            t = np.arange(tone_len) / sr
            tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

            # Push chunks via the callback (which queues them for the worker)
            for i in range(0, tone_len - hop, hop):
                chunk = tone[i:i + hop]
                indata = chunk.reshape(-1, 1).astype(np.float32)
                capture._audio_callback(
                    indata, hop, _MockTimeInfo(i / sr), 0
                )

            # Let the worker process
            time.sleep(0.3)

            notes = capture.get_notes()
            assert len(notes) >= 1, (
                f"Expected notes from worker, got {len(notes)}"
            )
            for tn in notes:
                assert tn.timestamp_ms >= 0.0
                assert tn.note.frequency > 0
        finally:
            capture.stop()

    def test_portable_worker_emits_notes_no_engine(self):
        """In portable mode there must be no PitchEngine — the worker
        calls PitchDetector.process directly.
        """
        config = Config()
        config.audio.profile = "portable"
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0

        capture = AudioCapture(config)
        assert capture._engine is None

        # Feed some audio
        sr = 44100
        hop = 512
        t = np.arange(sr * 0.3) / sr
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        capture._start_unified_worker()
        try:
            for i in range(0, len(tone) - hop, hop):
                chunk = tone[i:i + hop]
                indata = chunk.reshape(-1, 1).astype(np.float32)
                capture._audio_callback(
                    indata, hop, _MockTimeInfo(i / sr), 0
                )

            time.sleep(0.3)

            notes = capture.get_notes()
            assert len(notes) >= 1, "Expected notes from portable worker"
        finally:
            capture.stop()

    def test_high_accuracy_worker_emits_stable_events(self):
        """Feed 440 Hz through high_accuracy profile and expect notes.

        The worker should submit chunks to PitchEngine, drain candidates,
        and emit stable events via the stabilizer.
        """
        config = Config()
        config.audio.profile = "high_accuracy"
        config.audio.confidence_threshold = 0.3
        config.audio.noise_gate_db = -80.0

        capture = AudioCapture(config)
        engine = capture._engine
        assert engine is not None

        capture._start_unified_worker()
        try:
            sr = 44100
            hop = engine.hop_size
            t = np.arange(sr * 0.5) / sr
            tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

            for i in range(0, len(tone) - hop, hop):
                chunk = tone[i:i + hop]
                indata = chunk.reshape(-1, 1).astype(np.float32)
                capture._audio_callback(
                    indata, hop, _MockTimeInfo(i / sr), 0
                )

            time.sleep(0.4)

            notes = capture.get_notes()
            assert len(notes) >= 1, (
                f"Expected notes from high_accuracy worker, got {len(notes)}"
            )
        finally:
            capture.stop()
