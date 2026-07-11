# PickHero AppImage Packaging

This directory contains the files needed to build a Linux AppImage for PickHero.

## Prerequisites

```bash
pip install python-appimage
```

## Build

```bash
python-appimage build app -p 3.12 packaging/appimage/
```

Output: `PickHero-x86_64.AppImage`

## Run

```bash
chmod +x PickHero-x86_64.AppImage
./PickHero-x86_64.AppImage
```

## How it works

The AppImage bundles Python 3.12, all dependencies (from `requirements.txt`),
and the PickHero source. The `entrypoint.sh` sets `SDL_VIDEODRIVER=wayland`
for native Wayland rendering (falls back to X11 automatically if Wayland
is unavailable).

## Dependencies

All PickHero dependencies ship manylinux2014 wheels on PyPI:
- `aubio` (manylinux2014 x86_64)
- `pygame` (manylinux)
- `sounddevice` (manylinux with bundled PortAudio)
- `numpy` (manylinux)
- `pyguitarpro` (pure Python)

No build-from-source required.

## Fallback

If `python-appimage` fails to build (version incompatibility, missing
manylinux base), fall back to a shell script wrapper + venv approach:

```bash
python -m venv pickhero-venv
source pickhero-venv/bin/activate
pip install -r requirements.txt
python -m pickhero
```

Package as a `.tar.gz` with an install script.
