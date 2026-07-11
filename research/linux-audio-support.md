# Linux Audio & Packaging Support for PickHero

## Executive Summary

PickHero uses `sounddevice` (which wraps PortAudio) for audio capture and `pygame` (which wraps SDL2) for rendering. This stack maps cleanly onto modern Linux audio infrastructure: **PipeWire** is the recommended audio server for 2026 Linux desktops, providing transparent ALSA/JACK/PulseAudio emulation so PickHero works with zero code changes [1][2]. For packaging, **AppImage** via `python-appimage` is the simplest path for a Python+pygame app — single-file, no install, no sandbox to fight with for audio device access [3][4]. **Flatpak** is viable but the sandbox restricts direct ALSA access, requiring `device=all` permissions and reliance on the portal/RTKit path for real-time scheduling [5][6]. Users targeting sub-10ms latency need real-time scheduling permissions: the `realtime-privileges` package (or manual `limits.conf`) grants `RLIMIT_RTPRIO`, letting PipeWire's audio thread run at `SCHED_FIFO 88` instead of the RTKit ceiling of 20 [7][8]. On the display side, pygame's SDL2 backend **defaults to X11** even on Wayland desktops, routing through XWayland — setting `SDL_VIDEODRIVER=wayland` provides a superior experience but requires libdecor [9][10]. FMIT and Guitarix offer instructive patterns: FMIT's strategy-per-backend architecture (separate `CaptureThreadImpl` for ALSA, JACK, PortAudio, OSS, Qt) and Guitarix's JACK-native approach with real-time priority verification (`GxRtCheck`) are both relevant models [11][12][13].

PickHero's current code (`input.py`) already handles device name resolution, sample-rate matching, and ADC timestamp-based onset timing. The main gaps for Linux support are: (a) no `SDL_VIDEODRIVER` environment handling for Wayland, (b) no Linux packaging (only Windows PyInstaller spec exists), (c) device enumeration doesn't expose which PortAudio host API (ALSA vs JACK) a device belongs to, and (d) no documentation of real-time permission requirements for end users.

---

## Methodology

Sources searched: PortAudio documentation, PipeWire official docs, ArchWiki, NixOS Wiki, Flatpak Python docs, python-appimage docs, SDL2 wiki, pygame-ce GitHub issues, and source code of FMIT (`gillesdegottex/fmit`) and Guitarix (`brummer10/guitarix`). The PickHero codebase at `~/tmp/PickHero/` on branch `timing-judge` was read directly — `input.py`, `config.py`, `device_menu.py`, `main.py`, `app.py`, `pickhero.spec`, `pyproject.toml`, `requirements.txt`, and `.github/workflows/release.yml`. One pass, no sub-researchers spawned (depth-type research, 5 sub-questions, single domain). 13 primary sources cited.

---

## Findings

### 1. PipeWire vs JACK vs ALSA — Which Gives Best Results with sounddevice/PortAudio?

#### The PortAudio Host API Layer

PortAudio on Linux supports multiple host APIs: ALSA, JACK, OSS, PulseAudio, and sndio [2]. Each is a separate `PaHostApiRepresentation` implementation in `src/hostapi/alsa/pa_linux/alsa.c`, `src/hostapi/jack/pa_jack.c`, etc. PortAudio picks one as the default based on compile-time priority (ALSA is highest on Linux), but `Pa_GetHostApiCount()` exposes all available APIs [2].

PickHero's `sounddevice` library calls `sd.query_devices()` which wraps `Pa_GetDeviceCount()` + `Pa_GetDeviceInfo()` [14]. The current `list_audio_devices()` function at `input.py:246-260` iterates all devices and filters to `max_input_channels > 0`, but does **not** expose which host API each device belongs to. Each `PaDeviceInfo` struct contains a `hostApi` field (an index into the host API list) — sounddevice exposes this as `dev['hostapi']` in `query_devices()` results.

#### PipeWire: The Modern Default (Recommended)

PipeWire is the default audio server on Fedora (since 34), Ubuntu (since 22.04), Debian (since 12), and most other distributions as of 2026 [1]. It unifies ALSA, PulseAudio, and JACK into a single server:

- **`pipewire-alsa`**: provides an ALSA PCM plugin (`asym`/`dmix`-style) so ALSA applications transparently talk to PipeWire. PortAudio's ALSA host API connects through this — no code change needed.
- **`pipewire-pulse`**: PulseAudio replacement. SDL2's `pulseaudio` audio driver connects here.
- **`pipewire-jack`**: JACK replacement. Provides `libjack.so` shim so JACK applications connect to PipeWire without a real JACK daemon.

For PickHero via sounddevice/PortAudio ALSA backend: the audio path is `PortAudio ALSA → pipewire-alsa → PipeWire daemon → ALSA kernel driver → hardware`. This adds one hop compared to raw ALSA but provides automatic sample-rate conversion, device hotplugging, and per-client volume control [1][15].

**PipeWire latency configuration** (system-wide, affects all clients including PickHero):

```bash
# Create a drop-in config for low-latency defaults
mkdir -p ~/.config/pipewire/pipewire.conf.d/
cat > ~/.config/pipewire/pipewire.conf.d/99-low-latency.conf << 'EOF'
context.properties = {
    default.clock.rate = 44100
    default.clock.quantum = 256
    default.clock.min-quantum = 32
    default.clock.max-quantum = 1024
}
EOF

# Restart PipeWire to apply
systemctl --user restart pipewire pipewire-pulse wireplumber
```

The `quantum` parameter is the buffer size in samples — 256 samples at 44100Hz ≈ 5.8ms per cycle [15][16]. PickHero's current `hop_size=512` (medium preset) at 44100Hz gives ~11.6ms per process call, so a PipeWire quantum of 256 is well below the processing granularity. The `min-quantum=32` allows high-end USB interfaces to request smaller buffers.

**Disable device suspension** (prevents audio pops on USB interfaces that power down between captures):

```bash
mkdir -p ~/.config/wireplumber/wireplumber.conf.d/
cat > ~/.config/wireplumber/wireplumber.conf.d/50-disable-suspend.conf << 'EOF'
monitor.alsa.rules = [
    {
        matches = [
            { "node.name" = "~alsa_input.*" }
            { "node.name" = "~alsa_output.*" }
        ]
        actions = {
            update-props = {
                "session.suspend-timeout-seconds" = 0
            }
        }
    }
]
EOF
systemctl --user restart wireplumber
```

This is critical for PickHero: without it, the USB audio interface suspends between practice sessions, and the first note after resuming may be dropped or have a large timestamp error [15].

#### JACK: Lowest Latency, Most Setup

JACK provides the lowest round-trip latency (RTL) of the three — the NixOS guitar guide targets sub-6ms RTL with JACK/PipeWire at quantum 128/48kHz [15]. PortAudio's JACK host API (`pa_jack.c`) connects as a JACK client, mapping each JACK port to a PortAudio channel [2].

However, JACK requires running a JACK server (`jackd` or `pw-jack`), which adds operational complexity. On PipeWire systems, `pw-jack` transparently handles JACK clients, so the distinction is largely moot — PipeWire's JACK emulation provides near-identical latency to native JACK for capture-only use cases [1][17].

**To force PortAudio to use the JACK host API** (if a JACK server is running):

```bash
# Option 1: Set the default host API via PortAudio environment variable
# (Note: PortAudio does not have a direct env var for host API selection,
#  but sounddevice can query and select)

# Option 2: In Python, select a device from the JACK host API
python3 -c "
import sounddevice as sd
hostapis = sd.query_hostapis()
for i, api in enumerate(hostapis):
    print(f'[{i}] {api[\"name\"]}: devices={api[\"devices\"]}')
    # Typical output on PipeWire:
    # [0] ALSA: devices=[0, 1, 2, ...]   ← PipeWire ALSA devices
    # [1] JACK: devices=[...]             ← PipeWire JACK devices (if pw-jack active)
"
```

PickHero's `_resolve_device()` at `input.py:143-168` resolves by device name substring match, preferring mono inputs. It does **not** filter by host API. To prefer JACK devices:

```python
# Enhancement to _resolve_device: allow host API preference
def _resolve_device(self) -> int | None:
    ac = self.config.audio
    hostapis = sd.query_hostapis()
    # Prefer JACK host API if available (lowest latency)
    jack_api = next((a for a in hostapis if a['name'] == 'JACK'), None)
    # ... fall back to ALSA ...
```

#### ALSA Direct: Bypass the Server

Running PortAudio's ALSA backend against a real ALSA device (no PipeWire/PulseAudio) gives the most direct hardware path but loses mixing, hotplug, and per-client control [1][17]. This is the "ultimate low-latency" setup mentioned by LinuxMusicians users, but it means no other application can use audio simultaneously [17].

```bash
# Kill PipeWire/PulseAudio for raw ALSA access (aggressive, not recommended for desktop use)
systemctl --user stop pipewire pipewire-pulse wireplumber

# PickHero would then see raw ALSA hw devices
# PortAudio ALSA backend connects directly: PortAudio → ALSA lib → kernel → hardware
```

**`PA_ALSA_PLUGHW=1`**: PortAudio's ALSA host API has an environment variable that controls whether it uses `plughw:` (with format conversion) or `hw:` (direct) ALSA devices. Setting it to `1` (default) allows sample-rate/format conversion, which helps when the device doesn't natively support 44100Hz. PickHero already handles this at `input.py:161-162` by reading `dev["default_samplerate"]` and updating `ac.sample_rate` to match the device.

#### Recommendation for PickHero

| Scenario | Backend | Latency | Complexity |
|---|---|---|---|
| Default (most users) | PipeWire ALSA emulation | ~6-12ms | Zero config |
| Low-latency practice | PipeWire with quantum=128 | ~3-6ms | One config file |
| Pro audio user with JACK | PipeWire JACK emulation | ~3-6ms | Install pw-jack |
| Absolute minimum | Raw ALSA (no server) | ~2-4ms | Kill audio server |

**PickHero needs no code change for PipeWire** — the ALSA backend in PortAudio transparently connects. The key improvement is documenting the PipeWire quantum config and adding host API awareness to device enumeration.

---

### 2. AppImage vs Flatpak for Python+pygame Apps

#### Current State: Windows-Only Packaging

PickHero currently ships only a Windows executable via PyInstaller (`pickhero.spec`) built on `windows-latest` in CI (`.github/workflows/release.yml`). The spec bundles `sounddevice` (PortAudio DLL), `aubio`, `pygame` (SDL2 DLLs), `numpy`, and `certifi` — all as native Windows binaries. There is no Linux build path.

#### AppImage (Recommended for PickHero)

`python-appimage` [3][4] creates self-contained Linux AppImages from a Python application + recipe folder. The recipe contains:

```
recipe/
├── application.xml      # AppStream metadata
├── application.desktop  # Desktop entry
├── application.png      # Icon
├── requirements.txt     # pip dependencies
└── entrypoint.sh        # Startup script
```

**Build command:**
```bash
pip install python-appimage
python-appimage build app -p 3.12 /path/to/recipe/
# Output: PickHero-x86_64.AppImage
```

**PickHero AppImage recipe** (`packaging/appimage/`):

`requirements.txt`:
```
pygame>=2.5.0
sounddevice>=0.4.6
aubio>=0.4.9
numpy>=1.24.0
pyguitarpro>=0.9
certifi
```

`entrypoint.sh`:
```bash
{{ python-executable }} -I ${APPDIR}/opt/python{{ python-version }}/bin/pickhero "$@"
```

The `-I` flag isolates the AppImage's Python from the user's `~/.local` site-packages and `PYTHONPATH`, preventing version conflicts [4]. This is important for PickHero because `aubio` and `numpy` have C extensions that must match the bundled Python's ABI.

`application.desktop`:
```ini
[Desktop Entry]
Type=Application
Name=PickHero
Comment=Open-source guitar practice app with real-time pitch detection
Exec=pickhero
Icon=application
Terminal=false
Categories=AudioVideo;Audio;Music;Education;
Keywords=guitar;practice;tab;pitch;tuner;
```

**Pros for PickHero:**
- Single file, no installation — `chmod +x PickHero.AppImage && ./PickHero.AppImage`
- Direct ALSA access (no sandbox) — the AppImage process has full system permissions
- Can set `SDL_VIDEODRIVER` and other env vars in `entrypoint.sh`
- Manylinux2014 base provides glibc compatibility with most distros [4]
- No runtime dependencies beyond FUSE (standard on all desktop Linux)

**Cons:**
- `aubio` must be available as a manylinux wheel — it is on PyPI (manylinux2014 x86_64) ✓
- `sounddevice` bundles PortAudio — the manylinux wheel includes `libportaudio.so` ✓
- No automatic updates (AppImageUpdate exists but is manual)
- No central store/discovery (vs. Flathub)

**Critical caveat:** `python-appimage` can only package applications whose dependencies are available as binary wheels or pure Python [4]. `aubio` ships manylinux2014 wheels. `pygame` ships manylinux wheels. `sounddevice` ships manylinux wheels with bundled PortAudio. `numpy` ships manylinux wheels. `pyguitarpro` is pure Python. **All dependencies are satisfied.** ✓

#### Flatpak

Flatpak packages apps in a sandboxed environment with explicit permissions [5][6]. For Python apps, `flatpak-builder` with the `simple` buildsystem runs `pip3 install --prefix=/app`:

```yaml
# org.pickhero.PickHero.yml
id: org.pickhero.PickHero
runtime: org.freedesktop.Platform
runtime-version: '24.08'
sdk: org.freedesktop.Sdk
command: pickhero
finish-args:
  # X11/Wayland display
  - --share=ipc
  - --socket=fallback-x11
  - --socket=wayland
  # Audio — PulseAudio socket (PipeWire provides this)
  - --socket=pulseaudio
  # OR device=all for direct ALSA access
  - --device=all
  # Network for Songsterr downloads
  - --share=network
  # Config persistence
  - --filesystem=~/.pickhero:create
modules:
  - python3-requirements.json  # Generated by flatpak-pip-generator
  - name: pickhero
    buildsystem: simple
    build-commands:
      - pip3 install --prefix=/app --no-deps .
    sources:
      - type: dir
        path: .
```

Generate the Python dependencies manifest:
```bash
# Install flatpak-pip-generator
pip install flatpak-pip-generator

# Generate from requirements.txt
flatpak-pip-generator --requirements-file=requirements.txt --output=python3-requirements
```

This produces `python3-requirements.json` with each dependency as a separate Flatpak module, including tarball URLs and SHA256 hashes [5].

**Audio permissions — the critical issue:**

Flatpak's sandbox restricts audio device access. There are two paths [5][6]:

1. **`--socket=pulseaudio`**: The app connects via the PulseAudio socket. PipeWire provides this via `pipewire-pulse`. This is the "safe" path — the sandbox sees a PulseAudio server, not raw ALSA. SDL2's `pulseaudio` audio driver works here. **But sounddevice/PortAudio's ALSA backend does NOT work through this socket** — it needs raw ALSA device access.

2. **`--device=all`**: Grants access to all devices including `/dev/snd/*`. This allows PortAudio's ALSA backend to open raw ALSA PCM devices. This is the **required** permission for PickHero's sounddevice stack. However, this is a broad permission that some Flatpak reviewers (e.g., Flathub) are reluctant to grant.

3. **Portal Realtime**: Flatpak apps cannot use `RLIMIT_RTPRIO` directly (no `setrlimit` in the sandbox). They must use the xdg-desktop-portal Realtime interface, which delegates to RTKit with its priority-20 ceiling [7]. This means **Flatpak-packaged PickHero cannot achieve SCHED_FIFO 88** — it's capped at 20, which is sufficient for quantum 256+ but not for aggressive low-latency settings.

**Pros for PickHero:**
- Flathub distribution = discoverability + automatic updates
- Sandboxed = security
- Works on any distro with Flatpak (very broad coverage)

**Cons:**
- Audio device access requires `--device=all` (broad permission)
- No direct ALSA without `--device=all` — `--socket=pulseaudio` alone doesn't help sounddevice
- Real-time scheduling capped at RTKit priority 20 (no rlimits in sandbox)
- More complex build (flatpak-pip-generator, JSON manifests, SDK runtime)

#### Recommendation

**AppImage for v1**: simpler build, full audio access, no sandbox issues. Add Flatpak later if Flathub distribution is desired. The AppImage `entrypoint.sh` can also set `SDL_VIDEODRIVER` and other env vars, solving the Wayland issue (Section 4).

---

### 3. Real-Time Audio Permissions on Linux

PickHero's audio thread runs in the sounddevice callback (PortAudio's callback thread). PortAudio does not set real-time scheduling by itself on Linux — it relies on the system to grant `RLIMIT_RTPRIO` or uses RTKit/portal if available [7][8].

#### The Three Mechanisms

PipeWire (and JACK) acquire real-time scheduling through one of three mechanisms [7]:

| Mechanism | Max SCHED_FIFO | Requires | Latency Suitability |
|---|---|---|---|
| **RTKit** (D-Bus) | 20 (default) | `rtkit-daemon` running | Moderate (quantum ≥256) |
| **Portal Realtime** | 20 (delegates to RTKit) | `xdg-desktop-portal` | Moderate (quantum ≥256) |
| **rlimits** (PAM limits) | Up to 99 | `limits.conf` + group | Aggressive (quantum 64-128) |

**What a user needs:**

##### Method A: Install `realtime-privileges` (Easiest — Arch/Fedora)

```bash
# Arch Linux
sudo pacman -S realtime-privileges
sudo usermod -aG realtime $USER

# Fedora
sudo dnf install realtime-privileges
sudo usermod -aG realtime $USER

# Log out and back in for group membership to take effect
# Verify:
ulimit -r    # Should show 98 (Arch) or 95 (Fedora)
ulimit -l    # Should show unlimited
```

The `realtime-privileges` package installs `/etc/security/limits.d/99-realtime-privileges.conf` [8]:
```
@realtime - rtprio 98
@realtime - memlock unlimited
@realtime - nice -11
```

##### Method B: Manual `limits.conf` (Any distro)

```bash
# /etc/security/limits.d/audio.conf
@audio   -  rtprio     95
@audio   -  memlock    unlimited
@audio   -  nice       -19
```

```bash
sudo usermod -aG audio $USER
# Log out and back in
ulimit -r    # Should show 95
ulimit -l    # Should show unlimited
```

##### Method C: systemd user unit override (Cleanest — no PAM dependency)

```bash
mkdir -p ~/.config/systemd/user/pipewire.service.d/
cat > ~/.config/systemd/user/pipewire.service.d/realtime.conf << 'EOF'
[Service]
LimitRTPRIO=95
LimitMEMLOCK=infinity
LimitNICE=-19
EOF

systemctl --user daemon-reload
systemctl --user restart pipewire pipewire-pulse wireplumber
```

This is cleaner because it doesn't depend on PAM, doesn't require group membership, and applies only to PipeWire [7].

#### Verification

```bash
# Check which mechanism PipeWire is using
journalctl --user -u pipewire -b | grep -i "rt prio\|realtime\|rtkit\|rlimit"
# Expected: "acquired SCHED_FIFO priority 88" (rlimits) or
#           "acquired realtime priority 20" (RTKit)

# Check PipeWire's actual thread priority
ps -eLo pid,tid,cls,rtprio,ni,comm | grep pipewire
# Look for "FF" (SCHED_FIFO) in the cls column and a number in rtprio

# Check your own limits
ulimit -r    # RTPRIO limit (0 = no realtime)
ulimit -l    # MEMLOCK limit (should be "unlimited")
```

#### PipeWire's `module-rt` Configuration

PipeWire's `libpipewire-module-rt` [18] handles real-time scheduling automatically. Its default configuration in `client.conf`:
```
context.modules = [
    { name = libpipewire-module-rt
      args = {
          #nice.level = 20
          #rt.prio = 88
          #rt.time.soft = -1
          #rt.time.hard = -1
          #rlimits.enabled = true
          #rtportal.enabled = true
          #rtkit.enabled = true
      }
      flags = [ ifexist nofail ]
    }
]
```

The `rlimits.enabled = true` (default) means PipeWire tries rlimits first. If `RLIMIT_RTPRIO` is 0 (no permission), it falls back to `rtportal.enabled` (portal), then `rtkit.enabled` (RTKit) [7][18]. The `nofail` flag means if all three fail, PipeWire runs without real-time scheduling — audio works but with potential XRuns at low latencies.

#### What PickHero Specifically Needs

PickHero's audio callback (`input.py:57-79`) runs in PortAudio's callback thread. PortAudio's ALSA host API does **not** request real-time scheduling itself — it relies on the system scheduling the thread normally. The callback runs at whatever priority the kernel assigns.

For PickHero's current latency presets (`config.py:37-41`):
- **Low** (1024/256): ~12ms — works fine without RT scheduling at quantum 256+
- **Medium** (2048/512): ~23ms — never needs RT scheduling
- **High** (4096/1024): ~46ms — definitely never needs RT

**Real-time scheduling is not required for PickHero's default settings.** It becomes relevant only if users lower PipeWire's quantum below 256 or use the "low" preset with a high CPU load. The Timing Judge feature's timestamp accuracy depends on `time_info.inputBufferAdcTime` (already implemented at `input.py:67-71`), not on RT scheduling — RT prevents XRuns (dropped buffers) which would cause timestamp gaps, but the timestamps themselves are correct as long as the callback fires.

**Recommendation**: Document the `realtime-privileges` package in README as optional (for users who want the "low" latency preset), but do not require it. PickHero works correctly at medium/high presets without any RT configuration.

---

### 4. pygame on Wayland vs X11

#### The Problem

PickHero's `app.py:50-53` calls:
```python
pygame.init()
flags = pygame.RESIZABLE | pygame.SCALED
surface = pygame.display.set_mode((dc.width, dc.height), flags, vsync=1)
```

SDL2 (which pygame wraps) **defaults to the X11 video driver** on Linux, even when running on a Wayland desktop [9][10]. This means pygame applications run through XWayland (X11 compatibility layer) rather than native Wayland. The SDL2 wiki explicitly lists `x11` as the default and `wayland` as opt-in via `SDL_VIDEODRIVER=wayland` [10].

SDL3 (not yet used by pygame) changes this: it defaults to Wayland when the compositor supports the `fifo-v1` protocol [9]. But pygame currently uses SDL2, so the default is X11.

#### Known Issues with X11-on-Wayland

1. **Blurriness**: XWayland may not let legacy apps scale themselves, causing blurry rendering on HiDPI displays [9].
2. **Touch input bugs**: SDL2's X11 backend has major touch bugs [9].
3. **Tearing**: X11 driver can cause tearing in blits [11].
4. **No native Wayland features**: Fractional scaling, idle inhibition, and screen sharing don't work through XWayland.

#### The Fix: `SDL_VIDEODRIVER=wayland`

```bash
# Run PickHero with native Wayland
SDL_VIDEODRIVER=wayland python -m pickhero
```

Or set it in the AppImage entrypoint:
```bash
# entrypoint.sh
export SDL_VIDEODRIVER=wayland
{{ python-executable }} -I ${APPDIR}/opt/python{{ python-version }}/bin/pickhero "$@"
```

**Requirements**: `libdecor` must be installed (provides window decorations on Wayland). Most modern distros include it. If libdecor is missing, SDL2 falls back to X11 [9].

#### Recommended Code Change for PickHero

Add environment detection in `main.py` before `pygame.init()`:

```python
import os

def _configure_sdl_env():
    """Prefer native Wayland if available, fall back to X11."""
    if sys.platform.startswith("linux"):
        wayland_display = os.environ.get("WAYLAND_DISPLAY")
        if wayland_display and not os.environ.get("SDL_VIDEODRIVER"):
            # Don't override if user explicitly set it
            os.environ["SDL_VIDEODRIVER"] = "wayland"
```

This auto-detects Wayland sessions (`WAYLAND_DISPLAY` is set by Wayland compositors) and sets the SDL video driver only if the user hasn't explicitly set it. This is the pattern used by SDL3 internally and recommended by the SDL community [9].

**Fallback**: If `SDL_VIDEODRIVER=wayland` fails (e.g., missing libdecor), SDL2 automatically falls back to X11. So setting it is safe — worst case, it's ignored.

#### pygame-ce vs pygame

The pygame community edition (`pygame-ce`) has more active Wayland support [12]. PickHero's `requirements.txt` specifies `pygame>=2.5.0`, which could resolve to either `pygame` or `pygame-ce` (they're drop-in compatible). If Wayland issues arise, switching to `pygame-ce` explicitly may help:

```
# requirements.txt (alternative)
pygame-ce>=2.5.0
```

#### Audio Driver (SDL_AUDIODRIVER)

PickHero doesn't use SDL for audio (it uses sounddevice/PortAudio for capture and `midi_playback.py` for MIDI playback). But if MIDI playback uses pygame.midi, SDL2's audio subsystem initializes with `pygame.init()`. Setting `SDL_AUDIODRIVER=pipewire` ensures SDL2's audio connects to PipeWire rather than PulseAudio [10]:

```bash
export SDL_AUDIODRIVER=pipewire  # or pulseaudio (PipeWire provides pulse compat)
```

---

### 5. How FMIT and Guitarix Handle Linux Audio Device Enumeration

#### FMIT (Free Music Instrument Tuner)

FMIT (`gillesdegottex/fmit`) uses a **strategy pattern** with separate `CaptureThreadImpl` subclasses for each audio backend [11][13]:

| File | Backend | Approach |
|---|---|---|
| `CaptureThreadImplALSA.cpp` | ALSA | `snd_pcm_open()` + `snd_pcm_readi()` in a QThread |
| `CaptureThreadImplJACK.cpp` | JACK | `jack_client_open()` + `jack_port_register()` + ringbuffer |
| `CaptureThreadImplPortAudio.cpp` | PortAudio | `Pa_OpenStream()` + callback |
| `CaptureThreadImplOSS.cpp` | OSS | File descriptor I/O on `/dev/dsp*` |
| `CaptureThreadImplQt.cpp` | Qt Multimedia | `QAudioInput` |

The `CaptureThread` class (in `CaptureThread.h`) holds a `vector<CaptureThreadImpl*> m_impls` and a `m_current_impl` pointer. The `autoDetectTransport()` slot probes each implementation's `is_available()` method and picks the first one that works [13].

**FMIT's ALSA device enumeration** (`CaptureThreadImplALSA.cpp`):
- The ALSA source is a string like `"default"` or `"hw:0"`, set via `setSource()`.
- `is_available()` tries `snd_pcm_open()` with `SND_PCM_NONBLOCK` and catches errors (`-19` = invalid source, `-16` = device busy).
- Format negotiation: tries `SND_PCM_FORMAT_S16`, then `U16`, `S8`, `U8` in sequence.
- Sample rate negotiation: if set to `SAMPLING_RATE_MAX`, tries 8000→44100 in ascending order until one succeeds.
- Channel count: `snd_pcm_hw_params_set_channels_near()` to get closest to 1 channel, warns if >1.

**FMIT's PortAudio device enumeration** (`CaptureThreadImplPortAudio.cpp`):
- Calls `Pa_GetDeviceCount()`, iterates `Pa_GetDeviceInfo(i)`.
- Matches device by name: `if(QString(deviceInfo->name)==getASCIISource())`.
- If no match, falls back to `Pa_OpenDefaultStream()` (system default).
- Uses `paFloat32` format, 1 channel, `suggestedLatency = 0` (let PortAudio choose).
- **Notable**: FMIT's PortAudio callback receives `timeInfo` but explicitly casts it to `(void)` — it ignores the ADC timestamp! This is the same bug PickHero just fixed in the Timing Judge work.

**FMIT's JACK device handling** (`CaptureThreadImplJACK.cpp`):
- `jack_client_open()` with `JackNoStartServer` (doesn't auto-start JACK daemon).
- Registers an input port: `jack_port_register(client, "input", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0)`.
- Uses a `jack_ringbuffer_t` (1 second of audio) to decouple the JACK process callback from the capture thread.
- Auto-connects the source port if specified: `jack_connect(client, source, dest)`.
- Sample rate comes from the JACK server: `jack_get_sample_rate(client)` — the app cannot set it.

**Key pattern for PickHero**: FMIT's multi-backend strategy allows the user to select ALSA, JACK, or PortAudio from a dropdown. PickHero uses only PortAudio (via sounddevice), which is simpler but means the user can't choose between ALSA-direct and PipeWire-emulated-ALSA. Since PipeWire emulates ALSA transparently, this is fine for most users — but adding host API awareness to PickHero's device list would let users see which backend each device uses.

#### Guitarix

Guitarix (`brummer10/guitarix`) is **JACK-native** — it uses libjack directly, with no ALSA or PortAudio fallback [12]. The `GxJack` class (in `gx_jack.h`) manages:

- **Two JACK clients**: `client` (main processing) and `client_insert` (insert path), supporting single-client and dual-client modes.
- **Port registration**: `JackPorts` struct holds `input`, `midi_input`, `insert_out`, `midi_output`, `insert_in`, `output1`, `output2` — each a `PortConnection` with a `jack_port_t*` and a list of connected port names.
- **Callbacks**: `gx_jack_process` (audio processing), `gx_jack_insert_process` (insert path), `gx_jack_portreg_callback` (port registration), `gx_jack_portconn_callback` (port connection changes), `gx_jack_srate_callback` (sample rate), `gx_jack_buffersize_callback` (buffer size), `gx_jack_xrun_callback` (XRun notification).
- **Transport**: Tracks `jack_transport_state_t` and `jack_position_t` for timeline sync.
- **Session**: Optional JACK session support (`jack_session_callback`) for saving/restoring connections.
- **Real-time check**: The `GxRtCheck` class runs a background thread that verifies the user has real-time scheduling priority. If `set_priority()` fails, `IS_RT` is false and Guitarix warns the user.

**Guitarix's device/connection model**: Guitarix doesn't enumerate audio devices at all — it registers JACK ports and lets the user connect them via a patchbay (QjackCtl, Catia, qpwgraph). The `PortConnRing` class tracks connection changes via the JACK port connection callback and pushes them to the GUI via `Glib::Dispatcher` (thread-safe signal mechanism).

**Key pattern for PickHero**: Guitarix's `GxRtCheck` is directly relevant — PickHero could add a similar check at startup that verifies `ulimit -r > 0` and warns the user if real-time scheduling is not available, suggesting they install `realtime-privileges`. This would be especially useful when the user selects the "low" latency preset.

---

### PickHero Codebase Integration Points

#### Current Device Handling (Verified in Code)

PickHero's `list_audio_devices()` (`input.py:246-260`):
```python
def list_audio_devices() -> list[dict]:
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return inputs
```

This does **not** include the `hostapi` field. To add host API awareness:

```python
def list_audio_devices() -> list[dict]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            api_name = hostapis[dev["hostapi"]]["name"] if "hostapi" in dev else "unknown"
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
                "host_api": api_name,  # "ALSA", "JACK", etc.
            })
    return inputs
```

This lets the device menu (`device_menu.py`) show "[ALSA] USB Audio Interface" vs "[JACK] system" so users understand which backend they're using.

#### Environment Setup (Missing — Add to `main.py`)

PickHero's `main.py` has no environment configuration before `pygame.init()`. Adding the Wayland detection:

```python
def _configure_sdl_env():
    """Prefer native Wayland if available."""
    if sys.platform.startswith("linux"):
        if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("SDL_VIDEODRIVER"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
```

Called at the top of `main()` before any pygame import.

#### CI/Release (Missing Linux Build)

The `.github/workflows/release.yml` only builds on `windows-latest`. To add a Linux AppImage build:

```yaml
  build-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install python-appimage
      - run: python-appimage build app -p 3.12 packaging/appimage/
      - uses: softprops/action-gh-release@v2
        with:
          files: PickHero-x86_64.AppImage
```

---

## Sources

1. Linux Audio Quality Guide — ALSA, JACK, PipeWire Tuning — https://www.linuxdj.com/audio/quality/ (retrieved 2026-07-03)
2. PortAudio Unix/Linux Host APIs — https://deepwiki.com/PortAudio/portaudio/3.2-unixlinux-host-apis (retrieved 2026-07-03)
3. python-appimage Documentation — https://python-appimage.readthedocs.io/en/latest/ (retrieved 2026-07-03)
4. python-appimage Developers' Corner — https://python-appimage.readthedocs.io/en/latest/apps/ (retrieved 2026-07-03)
5. Flatpak Python Documentation — https://docs.flatpak.org/en/latest/python.html (retrieved 2026-07-03)
6. KDE Developer: Publishing your Python app as a Flatpak — https://develop.kde.org/docs/getting-started/python/python-flatpak/ (retrieved 2026-07-03)
7. PipeWire Real Time Scheduling in 2026: RTKit, Portal Realtime, and rlimits — https://www.linuxdj.com/notes/pipewire-real-time-scheduling-in-2026-rtkit-portal-and-rlimits/ (retrieved 2026-07-03)
8. ArchWiki: Realtime process management — https://wiki.archlinux.org/title/Realtime_process_management (retrieved 2026-07-03)
9. SDL2 should default to "wayland" video driver #11748 — https://github.com/libsdl-org/SDL/issues/11748 (retrieved 2026-07-03)
10. SDL2 Wiki: FAQ Using SDL — https://wiki.libsdl.org/SDL2/FAQUsingSDL (retrieved 2026-07-03)
11. How to avoid tearing with pygame on Linux/X11 — https://stackoverflow.com/questions/1082562/how-to-avoid-tearing-with-pygame-on-linux-x11 (retrieved 2026-07-03)
12. pygame-ce Issue #2058: PyGame seems to be running through X11 on top of wayland — https://github.com/pygame-community/pygame-ce/issues/2058 (retrieved 2026-07-03)
13. FMIT Source Code: `CaptureThread.h`, `CaptureThreadImplALSA.cpp`, `CaptureThreadImplJACK.cpp`, `CaptureThreadImplPortAudio.cpp` — https://github.com/gillesdegottex/fmit/tree/master/src (retrieved 2026-07-03)
14. PortAudio: Enumerating and Querying Devices — https://portaudio.com/docs/v19-doxydocs/querying_devices.html (retrieved 2026-07-03)
15. NixOS Wiki: Electric guitar interface setup — https://wiki.nixos.org/wiki/Electric_guitar_interface_setup (retrieved 2026-07-03)
16. PipeWire Configuration Index — https://docs.pipewire.org/page_config.html (retrieved 2026-07-03)
17. LinuxMusicians: Pipewire, Jack Applications & Low-Latency tuning — https://linuxmusicians.com/viewtopic.php?t=25556 (retrieved 2026-07-03)
18. PipeWire: RT Module Documentation — https://docs.pipewire.org/page_module_rt.html (retrieved 2026-07-03)
19. Guitarix Source Code: `gx_jack.h` — https://github.com/brummer10/guitarix/blob/master/trunk/src/headers/gx_jack.h (retrieved 2026-07-03)

## Confidence & Gaps

- **PipeWire as default audio server** — High confidence. Verified across NixOS Wiki [15], ArchWiki [8], PipeWire docs [16][18], and community sources [1][17]. Multiple independent sources agree.
- **PipeWire quantum/latency relationship** — High confidence. NixOS Wiki provides specific numbers (quantum 128 @ 48kHz ≈ 5ms) [15], corroborated by linuxdj.com [7].
- **AppImage recipe for PickHero** — Medium confidence. The recipe structure is from python-appimage docs [3][4]. Whether `aubio`'s manylinux wheel includes all required ALSA/PortAudio shared libraries at runtime has not been verified by building. The `sounddevice` wheel bundles `libportaudio.so` — verified from PyPI. `aubio` wheel bundles its own `.so` — likely but not confirmed in this session. `[INFERENCE]` on aubio's wheel contents.
- **Flatpak `--device=all` requirement for sounddevice** — Medium confidence. Based on Flatpak permission model [5][6] and PortAudio's ALSA backend architecture [2]. Not tested by building a Flatpak of PickHero. The `--socket=pulseaudio` path definitely doesn't help PortAudio's ALSA host API — this is architectural, not empirical. `[INFERENCE]` that `--device=all` is the only viable path.
- **RTKit priority ceiling of 20** — High confidence. Explicitly stated in the linuxdj.com guide [7] and PipeWire module-rt docs [18].
- **SDL2 defaults to X11** — High confidence. SDL2 wiki [10] lists x11 as default. GitHub issue #11748 [9] requests changing the default. SDL3 changes this [9].
- **FMIT's PortAudio callback ignores timeInfo** — High confidence. Read directly from source code (`CaptureThreadImplPortAudio.cpp`): `(void)timeInfo;` [13].
- **Guitarix is JACK-only** — High confidence. Read directly from `gx_jack.h` — no ALSA or PortAudio includes, only `#include <jack/jack.h>` [19]. The build system conditionally compiles with `GUITARIX_AS_PLUGIN` for DAW plugin mode.
- **Flatpak real-time scheduling capped at RTKit 20** — Medium-high confidence. The linuxdj.com guide [7] states that portal realtime delegates to RTKit (ceiling 20) and that rlimits don't work in Flatpak. This is architecturally sound but not empirically tested with PickHero.
- **Missing**: Empirical latency measurements of PickHero on PipeWire vs raw ALSA. The NixOS guide [15] provides benchmarks for a different audio chain (guitar processing, not pitch detection). PickHero's latency is dominated by the onset detector's algorithmic delay (2201 samples ≈ 50ms at 44100Hz), not by the audio backend — so backend choice has minimal practical impact on perceived latency for this application. `[INFERENCE]`
