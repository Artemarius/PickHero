# PickHero — CLAUDE.md

## Project Overview

Desktop guitar practice app. Scrolling Guitar Pro tabs (GP3/GP4/GP5/GP6/GP7/GP8) with real-time pitch detection and visual hit/miss feedback. Python, PyGame, aubio, pyguitarpro. Must run on low-end hardware (no ML, no GPU).

## Language & Stack

- **Python 3.10+**, Windows primary target
- **aubio** for pitch detection (YIN algorithm) and onset detection
- **sounddevice** for audio capture from USB audio devices
- **pyguitarpro** for reading GP3/GP4/GP5 tab files; native XML/BCFZ parser for GP6/GP7/GP8
- **pygame** for UI rendering (scrolling display, game loop)
- **pygame.mixer** for backing track / metronome playback
- No ML frameworks. No TensorFlow, no CREPE, no PyTorch. Detection is signal-processing only.

## Architecture

Three threads:
1. **Audio thread** — `sounddevice` callback captures audio, feeds to aubio pitch/onset detectors, pushes detected notes to a thread-safe queue
2. **Main thread** — PyGame event loop, renders scrolling UI, reads detected notes from queue, runs note matcher against timeline
3. **Playback thread** (optional) — pygame.mixer for backing track audio

Modules:
- `pickhero/audio/` — capture, detection, note utilities. No UI dependencies.
- `pickhero/tabs/` — GP file loading, timeline data structure, Songsterr downloader. No UI dependencies.
- `pickhero/ui/` — PyGame rendering, game loop, menus. Depends on audio and tabs.
- `pickhero/config.py` — user settings (audio device, noise gate, visual prefs). JSON file in user home dir.

## Key Conventions

- **Module independence:** audio/ and tabs/ must be testable without PyGame. No pygame imports outside ui/.
- **Thread safety:** audio thread communicates with main thread via `queue.Queue`. No shared mutable state.
- **Note representation:** use MIDI note numbers internally (0-127). Convert to name/octave only for display.
- **Timing:** all timestamps in milliseconds (float). Timeline positions are ms from song start.
- **Frequency → note:** use `round(12 * log2(freq / 440) + 69)` for MIDI note number. Standard A4 = 440 Hz.
- **Tolerance:** pitch match within ±1 semitone = "close" (yellow). Exact semitone = "hit" (green). Timing window configurable, default 100ms.
- **Guitar tuning:** standard E2-A2-D3-G3-B3-E4 (MIDI 40-45-50-55-59-64). Support alternate tunings from GP file header.

## aubio Configuration

```python
# Pitch detection
pitch_detector = aubio.pitch("yin", buf_size=2048, hop_size=512, samplerate=44100)
pitch_detector.set_unit("Hz")
pitch_detector.set_tolerance(0.8)  # confidence threshold

# Onset detection
onset_detector = aubio.onset("default", buf_size=2048, hop_size=512, samplerate=44100)
onset_detector.set_threshold(0.3)  # adjust based on testing
```

- Buffer/hop sizes tuned for guitar frequency range (82 Hz low E to ~1300 Hz high E 24th fret)
- 44100 Hz sample rate (standard for USB audio devices)
- Noise gate: ignore pitches below configurable confidence threshold

## Guitar Pro File Loading

GP3/GP4/GP5 files are read with pyguitarpro: iterate tracks → find guitar/bass track(s) → iterate measures → beats → notes.
GP6 (.gpx) files are BCFZ/BCFS containers that hold a `score.gpif` XML file; GP7/GP8 files are ZIP containers with the same XML. The native parser extracts that XML and builds the timeline.

```python
# Each note gives: note.value (fret), note.string (1-N), beat.start, beat.duration
# Convert to: (timestamp_ms, midi_note, string, fret, duration_ms)
```

Tempo changes: GP files can have tempo changes per measure. Track cumulative time, don't assume constant BPM.

## PyGame Rendering

- **Window:** 1280×720 default, resizable
- **Layout:** 6 horizontal lanes (one per string), notes scroll right-to-left
- **Note display:** rectangles on string lanes, width proportional to duration, fret number drawn on note
- **Scroll speed:** pixels_per_ms = lane_width / visible_window_ms. Derive from BPM.
- **Hit zone:** vertical line on left side of screen. Notes passing through it are "active" for matching.
- **Target FPS:** 60 (PyGame clock.tick)

## File Organization

Keep it flat and simple. Don't over-engineer packages:
```
pickhero/
├── __init__.py
├── __main__.py          # python -m pickhero entry point
├── main.py
├── config.py
├── matcher.py           # note matching engine (hit/close/miss)
├── progress.py          # per-song progress tracking
├── audio/
│   ├── __init__.py
│   ├── input.py
│   ├── detector.py
│   ├── console.py       # audio testing console (pitch/chord/synth/list subcommands)
│   ├── midi_playback.py
│   └── note_utils.py
├── tabs/
│   ├── __init__.py
│   ├── loader.py
│   ├── timeline.py
│   └── downloader.py
└── ui/
    ├── __init__.py
    ├── app.py
    ├── calibration_menu.py  # Guitar calibration wizard
    ├── colors.py          # Theme system (dark/light)
    ├── scrolling.py
    ├── feedback.py
    ├── menu.py
    ├── device_menu.py
    └── download_menu.py
```

## Testing

- `tests/test_detector.py` — feed known sine waves to aubio, verify correct note detection
- `tests/test_loader.py` — load reference GP5/GP6/GP7 files and verify extracted notes
- `tests/test_chord_detector.py`, `tests/test_console.py`, `tests/test_app.py`, `tests/test_integration.py` — detector, UI, console, and end-to-end smoke tests
- `tests/test_downloader.py` — Songsterr search/download with mocked urllib responses
- Use `pytest`. Keep tests independent of audio hardware (mock sounddevice).

## Build & Run

```bash
pip install -r requirements.txt
python -m pickhero

# Audio testing console (no GUI)
python -m pickhero console                    # live pitch detection
python -m pickhero console list               # list audio input devices
python -m pickhero console chord E2 A2 D3     # FFT chord verification
python -m pickhero console synth E2 A2 D3    # synthetic signal test

# Package for distribution
pip install pyinstaller
pyinstaller pickhero.spec --noconfirm
# Or use build.bat on Windows

## What NOT To Do

- Don't add ML-based pitch detection. aubio YIN is sufficient and runs everywhere.
- Don't create a web UI or Electron wrapper. This is a desktop app.
- Don't add online features, accounts, or cloud sync. Offline-first, local files only.
- Don't over-abstract. Simple classes, no deep inheritance hierarchies. This is a ~3K LOC app, not a framework.
- Don't add polyphonic pitch detection. aubio YIN is monophonic. Chord scoring uses a majority-match model instead.
