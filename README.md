# PickHero 🎸

Free, open-source desktop guitar practice app with real-time pitch detection and scrolling tab playback — a lightweight Yousician alternative.

## Why?

Yousician is great but expensive for continuous use, and there's no good free alternative that combines Guitar Pro tab playback with live pitch detection. PickHero fills that gap: load any GP3/GP4/GP5 tab file, plug in your guitar via a cheap USB cable, and practice with real-time visual feedback — all without an internet connection or subscription.

Designed to run on modest hardware (tested on an HP ProBook 650 G5 laptop). No ML models, no GPU required.

## How It Works

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│ Audio Input   │────▶│ Pitch Engine  │────▶│ Note Matcher     │
│ (sounddevice) │     │ (aubio YIN)   │     │ (compare to tab) │
└──────────────┘     └───────────────┘     └────────┬─────────┘
                                                     │
┌──────────────┐     ┌───────────────┐              │
│ Tab Loader   │────▶│ Tab Timeline  │◀─────────────┘
│ (PyGuitarPro)│     │ (note events) │
└──────────────┘     └───────┬───────┘
                             │
                     ┌───────▼───────┐
                     │ Scrolling UI  │
                     │ (PyGame)      │
                     │ + feedback    │
                     └───────────────┘
```

Guitar Pro tabs scroll across the screen while the app listens to your guitar input, detects the notes you play in real-time using the YIN pitch detection algorithm, and shows visual feedback (hit / miss / close) synchronized with the tab timeline.

## Hardware Setup

Electric guitar → USB guitar cable (1/4" TS to USB-A, ~€12-15) → PC. The cable appears as a standard "USB Audio Device" in Windows. For hearing yourself while playing, use a regular guitar amp alongside or split the signal — the USB cable is input-only for detection.

A regular microphone also works for acoustic guitar or as a quick test, though a direct USB connection gives cleaner detection.

## Tab Sources

PickHero reads Guitar Pro files (`.gp3`, `.gp4`, `.gp5`). You can get tabs from:

- **Songsterr** — 1M+ songs, GP5 download via built-in downloader
- **GProTab.net** — 70K+ free Guitar Pro files
- **TuxGuitar** — free editor for creating your own tabs
- **Ultimate Guitar** — GP files available (some require subscription)

## Tech Stack

| Component | Library | Why |
|---|---|---|
| Audio capture | `sounddevice` | Works with any USB audio device, low latency |
| Pitch detection | `aubio` (YIN) | Real-time, pure C, tiny footprint |
| Onset detection | `aubio` | Detects note strikes for timing |
| Tab parsing | `pyguitarpro` | Reads GP3/GP4/GP5 structured data |
| UI | `pygame` | Game-loop oriented, fast rendering |
| Audio playback | `pygame.midi` | Real-time MIDI backing tracks via system synth |
| Packaging | `PyInstaller` | Single .exe for Windows |

## Installation

```bash
# Clone
git clone https://github.com/Artemarius/PickHero.git
cd PickHero

# Install dependencies
pip install -r requirements.txt

# Run
python -m pickhero
```

### Requirements

- Python 3.10+
- Windows 10/11 (primary target; Linux/macOS may work but untested)
- A USB audio input device (guitar cable or microphone)

## Project Structure

```
PickHero/
├── pickhero/
│   ├── main.py              # Entry point
│   ├── audio/
│   │   ├── input.py         # sounddevice audio capture
│   │   ├── detector.py      # aubio pitch + onset detection
│   │   ├── midi_playback.py # MIDI backing track playback
│   │   └── note_utils.py    # frequency → note/string/fret mapping
│   ├── tabs/
│   │   ├── loader.py        # pyguitarpro file reader
│   │   ├── timeline.py      # song timeline data structure
│   │   └── downloader.py    # Songsterr tab fetcher
│   ├── ui/
│   │   ├── app.py           # main game loop / window
│   │   ├── scrolling.py     # scrolling note highway renderer
│   │   ├── colors.py        # color constants, string palette
│   │   ├── device_menu.py   # audio input device selector
│   │   ├── feedback.py      # hit/miss visual effects
│   │   └── menu.py          # song selection, settings
│   ├── config.py            # user settings, audio device config
│   └── progress.py          # per-song best score tracking
├── assets/
│   └── fonts/
├── songs/                   # local GP5 tab storage
└── tests/
```

## Status

**Under active development** — see [Development Phases](#development-phases) below.

### Development Phases

1. ~~**Audio Detection PoC** — `sounddevice` + `aubio` pitch detection, console output~~ **Done**
2. ~~**Tab Parser & Timeline** — GP5 loading via `pyguitarpro`, timeline data structure~~ **Done**
3. ~~**Scrolling Display MVP** — PyGame window with 6 string lanes, tempo-synced scrolling~~ **Done**
4. ~~**Live Matching & Feedback** — pitch comparison, hit/miss visuals, accuracy scoring~~ **Done**
5. **Polish** — ~~tempo control~~, ~~section looping~~, ~~device selector~~, ~~backing tracks~~, ~~count-in~~, ~~progress tracking~~, song browser, `.exe` packaging

## References

- [aubio](https://github.com/aubio/aubio) — pitch/onset detection
- [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) — GP file parser
- [TabRiPP](https://github.com/josipnigojevic/TabRiPP) — Songsterr GP5 downloader (reference for API approach)
- [AlphaTab](https://github.com/CoderLine/alphaTab) — JS tab rendering engine (UI reference)
- [TuxGuitar](https://tuxguitar.app/) — free Guitar Pro editor

## License

MIT
