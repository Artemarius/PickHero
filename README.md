# PickHero 🎸

Free, open-source desktop guitar practice app with real-time pitch detection and scrolling tab playback — a lightweight Yousician alternative.

## Why?

Yousician is great but expensive for continuous use, and there's no good free alternative that combines Guitar Pro tab playback with live pitch detection. PickHero fills that gap: load any Guitar Pro tab file (GP3/GP4/GP5/GP7/GP8), plug in your guitar via a cheap USB cable, and practice with real-time visual feedback — all without an internet connection or subscription.

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

PickHero reads Guitar Pro files (`.gp3`, `.gp4`, `.gp5`, `.gp7`, `.gp8`). You can get tabs from:

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

### Option 1: Download (recommended)

Grab `PickHero.exe` from the [latest release](https://github.com/Artemarius/PickHero/releases/latest). No Python install needed — just run it.

### Option 2: From source

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

- Windows 10/11 (primary target; Linux/macOS may work but untested)
- A USB audio input device (guitar cable or microphone)
- Python 3.10+ (only if running from source)

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
│   │   ├── calibration_menu.py # guitar calibration wizard
│   │   ├── device_menu.py   # audio input device selector
│   │   ├── feedback.py      # hit/miss visual effects
│   │   └── menu.py          # song selection, settings
│   ├── config.py            # user settings, audio device config
│   ├── matcher.py           # note matching engine (hit/close/miss)
│   ├── progress.py          # per-song score tracking + section history
│   └── recommendations.py   # practice suggestion engine
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
5. ~~**Polish** — tempo control, section looping, device selector, backing tracks, count-in, progress tracking, song browser, difficulty filter, themes, practice recommendations, `.exe` packaging~~ **Done**
6. ~~**Wait Mode** — beginner-friendly practice mode that pauses playback when you haven't played the correct note, resumes instantly when you do. Toggle with **W** during playback.~~ **Done**

## Troubleshooting

**No notes detected / no signal:**
- Turn up the volume knob on your guitar (most common gotcha!)
- Check the pickup selector is in a live position (bridge or neck, not between clicks)
- Unplug and re-plug the USB cable at both ends
- Use a rear motherboard USB port, not a front panel or hub
- In Windows: `Settings > System > Sound > Input` — select "USB Audio" and check the volume slider isn't at zero. You should see the input meter move when you strum.

**Testing your setup:**
Run `python -m pickhero --console` to see detected notes printed live in the terminal. This helps isolate whether the issue is hardware (no signal) or software (signal present but not matching).

**Audio detection not working in-game:**
- Audio detection is enabled by default. If you turned it off, press **A** during playback to re-enable it. The bottom status bar should show `A: audio ON`.
- Press **D** on the menu screen to select the correct audio input device.
- Press **G** on the menu screen to run guitar calibration — this improves detection accuracy by learning your guitar's actual frequencies.

**Beginner tip — Wait Mode:**
- Press **W** during playback to enable wait mode. The tab will pause whenever you haven't played the correct note yet and resume the moment you do. Great for learning new songs at your own pace without accumulating misses.

**High latency or missed notes:**
- Lower the noise gate with **X** during playback (default is -60 dB)
- Cheap USB cables can add 10-20ms of latency — this is usually fine but noticeable on fast passages
- Slow the song down with **PgDn** while practicing
- Press **H** during playback to see a full help overlay with controls and color legend

## References

- [aubio](https://github.com/aubio/aubio) — pitch/onset detection
- [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) — GP file parser
- [TabRiPP](https://github.com/josipnigojevic/TabRiPP) — Songsterr GP5 downloader (reference for API approach)
- [AlphaTab](https://github.com/CoderLine/alphaTab) — JS tab rendering engine (UI reference)
- [TuxGuitar](https://tuxguitar.app/) — free Guitar Pro editor

## License

MIT
