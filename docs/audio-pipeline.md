# Audio Pipeline

This document describes how audio flows through PickHero, where computation
happens, and how latency is modeled and compensated.

## Clock ownership

`StreamClock` in `pickhero/audio/clock.py` owns the mapping between the
monotonic capture stream and song time. The audio callback
(`input.py:_audio_callback`) must **never** call `StreamClock` methods directly;
it only copies the selected input channel into the ring buffer. The main loop
in `pickhero/ui/scrolling.py` is the only code that calls:

- `set_segment(...)` — when the player seeks, pauses, resumes, or changes tempo.
- `song_to_stream_ms(...)` — to find the capture position for a given song time.
- `stream_to_song_ms(...)` — to map a captured sample timestamp back to chart time.

Keeping clock state on the main thread avoids races with the realtime audio
callback.

## Callback constraints

`_audio_callback` in `pickhero/audio/input.py` does minimal work:

1. Select the configured input channel from the incoming multi-channel buffer.
2. Copy the frame into the ring buffer (`input.py:165`).

All DSP (pitch detection, chord verification, onset detection, etc.) runs on
the unified worker thread. The single `.copy()` in the callback is currently the
only heap allocation in the hot path. It is flagged for future optimization but
is acceptable for Phase 0.

## Latency compensation

`StreamClock._latency_offset_ms` offsets every song↔stream calculation. It is
applied through `config.set_audio_latency_offset()`. The full latency breakdown
is available from `AudioCapture.get_latency_breakdown()`, which returns:

- `reported_input_ms` — latency reported by the audio backend for the input device.
- `onset_detector_ms` — algorithmic delay introduced by the onset detector.
- `stabilizer_confirmation_ms` — time the note stabilizer waits before committing.
- `manual_or_loopback_trim_ms` — user-adjusted or measured loopback trim.

## Detector delay

`PitchEngine._SPECTRAL_BUF_SIZE = 16384` (`pitch_engine.py:103`) is the rolling
spectral-analysis window. At 48 kHz this buffers approximately 341 ms of audio
before spectral hypotheses are available.

YIN-based detection has lower latency: the configured `hop_size` of 512 samples
at 48 kHz is approximately 10.7 ms per analysis hop.
