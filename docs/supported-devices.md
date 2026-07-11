# Supported Devices

This document lists audio interfaces and backends tested with PickHero, along
with recommended settings and known caveats. The data is also available
programmatically in `pickhero/audio/device_matrix.py`.

## Latency presets

PickHero defines three latency presets:

| Mode    | Buffer | Hop  | Typical input latency |
| ------- | ------ | ---- | -------------------- |
| low     | 1024   | 256  | ~12 ms               |
| medium  | 2048   | 512  | ~23 ms               |
| high    | 4096   | 1024 | ~46 ms               |

Higher buffer sizes improve detection stability for low notes at the cost of
latency. The `low` preset may miss low register content (E2–A2 on guitar).

Detection profile (`portable` vs `high_accuracy`) also affects latency:
`high_accuracy` requires 48 kHz sample rate, 256 hop, 4096 buffer, and adds a
~341 ms spectral analysis window regardless of hop size.

---

## Linux (ALSA / PulseAudio / PipeWire)

sounddevice uses PortAudio on Linux. On modern distros PipeWire presents a
PulseAudio-compatible API, and PulseAudio in turn wraps ALSA. All three paths
work identically from PickHero's perspective.

### Tested USB interfaces

| Device              | Backend         | Sample rates | Buffer sizes            | Measured input latency          | Known issues                     |
| ------------------- | --------------- | ------------ | ----------------------- | ------------------------------- | -------------------------------- |
| Focusrite Scarlett Solo (3rd gen) | ALSA via PulseAudio | 44100, 48000 | 256, 512, 1024, 2048 | ~8–12 ms at 512/48k           | None observed with `low`/`medium` |
| Focusrite Scarlett 2i2 (3rd gen) | ALSA via PulseAudio | 44100, 48000 | 256, 512, 1024, 2048 | ~10–14 ms at 512/48k           | Power-on pop can saturate input briefly |
| Behringer UMC22     | ALSA via PulseAudio | 44100, 48000 | 512, 1024, 2048       | ~15–20 ms at 1024/48k         | 256 buffer may glitch; `high` preset recommended |
| Behringer UMC202HD  | ALSA via PulseAudio | 44100, 48000 | 512, 1024, 2048       | ~10–18 ms at 512/48k          | Needs `snd-usb-audio` quirks on older kernels |
| Built-in (HDA Intel) | ALSA via PulseAudio/PipeWire | 44100, 48000 | 512, 1024, 2048       | ~15–30 ms                      | Higher jitter; use USB interface for reliable detection |

### Recommended settings

- **USB interfaces** (any of the above): `medium` (2048/512) at 48 kHz for best
  detection quality; `low` (1024/256) at 48 kHz for a snappier feel when
  detection reliability is less critical.
- **Built-in audio**: `medium` (2048/512) at 48 kHz. Avoid `low` preset —
  built-in codecs add enough jitter that the smaller buffer causes dropouts.

### Known issues

- **PulseAudio default latency**: PortAudio may clamp requested latency if
  PulseAudio is configured with `default-fragments` or `default-fragment-size-msec`
  in `/etc/pulse/daemon.conf`. If you hear crackling, increase the latency preset
  or adjust PulseAudio's fragment size.
- **PipeWire quantum**: PipeWire's graph quantum may override PortAudio's
  buffer request. Set `default.clock.min-quantum` and `default.clock.max-quantum`
  in `/etc/pipewire/pipewire.conf` if you see unexpected glitches.
- **snd-usb-audio**: Some USB interfaces (especially Behringer) require the
  `device_setup` quirk on kernel 6.1–6.5. This is resolved on 6.6+.
- **Sample rate mismatch**: If `sounddevice` reports a different `default_samplerate`
  from what the device is physically clocked at, the stream may resample
  silently. Check `sd.query_devices()` and ensure the interface clock matches
  the sample rate in PickHero config.

---

## Windows (WASAPI / ASIO)

On Windows, sounddevice selects the backend in this order:

1. **ASIO** (if `audio.asio_enabled = True`) — exclusive access, lowest latency.
2. **WASAPI exclusive** (if ASIO unavailable or disabled) — low latency, but
   exclusive device access.
3. **WASAPI shared** — fallback, higher latency, but works without fighting
   other audio apps for the device.

### Tested USB interfaces

| Device                   | Backend       | Sample rates | Buffer sizes                    | Measured input latency           | Known issues                     |
| ------------------------ | ------------- | ------------ | ------------------------------- | -------------------------------- | -------------------------------- |
| Focusrite Scarlett Solo (3rd gen) | ASIO (Focusrite USB ASIO) | 44100, 48000 | ASIO: driver-chosen (~64–256); WASAPI: 256, 512, 1024, 2048 | ~4–8 ms (ASIO), ~10–14 ms (WASAPI) | ASIO channel selector must be set to input 1 |
| Focusrite Scarlett 2i2 (3rd gen) | ASIO (Focusrite USB ASIO) | 44100, 48000 | ASIO: driver-chosen; WASAPI: 256, 512, 1024, 2048 | ~4–8 ms (ASIO), ~10–14 ms (WASAPI) | 2i2 has two inputs; set `input_channel` to 0 or 1 |
| Behringer UMC22          | ASIO4ALL v2   | 44100, 48000 | ASIO4ALL: 256, 512, 1024; WASAPI: 512, 1024, 2048 | ~10–18 ms (ASIO4ALL), ~18–30 ms (WASAPI) | ASIO4ALL buffer below 256 causes crackling; `latency_mode="low"` with ASIO4ALL may fail |
| Behringer UMC202HD       | ASIO (Behringer USB ASIO) | 44100, 48000 | ASIO: 256, 512, 1024; WASAPI: 512, 1024, 2048 | ~8–14 ms (ASIO)              | Behringer ASIO driver only supports 44.1 kHz at 256 buffer minimum |
| Built-in Realtek (ALC892/ALC1220) | WASAPI shared | 44100, 48000 | 1024, 2048 | ~20–40 ms                     | WASAPI exclusive may not work on all Realtek codecs; use shared fallback |
| ASIO4ALL (generic WDM wrapper) | ASIO4ALL v2   | 44100, 48000 | 512, 1024, 2048             | ~12–25 ms                    | Inconsistent across WDM drivers; test each device pair |

### Recommended settings

- **Focusrite Scarlett (ASIO)**: ASIO with `medium` (2048/512) at 48 kHz.
  For minimal latency, use `low` (1024/256) at 48 kHz — the Focusrite ASIO
  driver handles small buffers well.
- **Behringer UMC22 (ASIO4ALL)**: `medium` (2048/512) at 48 kHz with
  `asio_buffer_size = 512`. Do not use `low` preset.
- **Behringer UMC202HD (native ASIO)**: `medium` (2048/512) at 48 kHz.
- **Built-in Realtek**: `high` (4096/1024) at 48 kHz, WASAPI shared.
  Latency with built-in audio makes PickHero playable but accuracy suffers;
  a USB interface is strongly recommended.

### Known issues

- **ASIO channel selector**: When `asio_enabled = True`, PickHero passes
  `sd.AsioSettings(channel_selectors=[input_channel])`. If the device has
  multiple inputs, set `input_channel` to the correct zero-based index.
  A wrong selector causes silence or capture of the wrong channel.
- **ASIO4ALL buffer negotiation**: PortAudio may report a different actual
  buffer size than what ASIO4ALL claims. If you hear glitches, raise
  `asio_buffer_size` in config (overrides `hop_size` for ASIO).
- **WASAPI exclusive**: While active, no other application can use the audio
  device. The system may also block sleep/hibernate. Switch to WASAPI shared
  if this is a problem.
- **Device name matching**: Win32 device names from `sd.query_devices()` may
  include backend prefixes like `"ASIO: Focusrite USB ASIO"`. The
  `device_matrix.py` matching strips these prefixes for fuzzy matching.

---

## macOS (CoreAudio)

On macOS sounddevice uses PortAudio's CoreAudio backend, which is well-tested
and reliable.

### Tested devices

| Device              | Backend   | Sample rates | Buffer sizes     | Measured input latency | Known issues                     |
| ------------------- | --------- | ------------ | ---------------- | ---------------------- | -------------------------------- |
| Focusrite Scarlett Solo (3rd gen) | CoreAudio | 44100, 48000 | 256, 512, 1024, 2048 | ~6–10 ms at 512/48k | None observed                    |
| Focusrite Scarlett 2i2 (3rd gen) | CoreAudio | 44100, 48000 | 256, 512, 1024, 2048 | ~6–10 ms at 512/48k | Occasional kernel extension (kext) approval needed on macOS < 12 |
| Behringer UMC22     | CoreAudio | 44100, 48000 | 512, 1024, 2048  | ~14–20 ms at 1024/48k | 256 buffer may glitch; `high` preset recommended |
| Behringer UMC202HD  | CoreAudio | 44100, 48000 | 512, 1024, 2048  | ~10–16 ms at 512/48k  | macOS may need USB audio class compliance mode |
| Built-in (MacBook Pro) | CoreAudio | 44100, 48000 | 512, 1024, 2048  | ~10–20 ms             | 3.5 mm jack has higher noise floor |

### Recommended settings

- **USB interfaces** (any Focusrite): `medium` (2048/512) or `low` (1024/256)
  at 48 kHz. CoreAudio handles both well.
- **Behringer UMC22**: `medium` (2048/512) at 48 kHz; avoid `low`.
- **Built-in audio**: `medium` (2048/512) at 48 kHz. The built-in jack is
  usable for practice but a USB interface improves detection reliability.

### Known issues

- **macOS permission**: On macOS 10.14+, PickHero must have microphone
  permission. If no audio is captured, check System Preferences > Security &
  Privacy > Microphone.
- **Kext approval** (Intel macOS < 12): Some Focusrite interfaces need
  Focusrite Control or the kernel extension approved. On Apple Silicon this
  is not required.
- **Sample rate changes**: Changing sample rate while the interface is used by
  another app (e.g. Focusrite Control, system sound) may cause the stream to
  open at the wrong rate. Close other apps using the interface before starting
  PickHero.

---

## Cross-platform notes

### Sample rate choice

- **48 kHz** is recommended for all interfaces on all platforms when using the
  `high_accuracy` profile (which requires `sample_rate >= 48000`). The
  `portable` profile works at either 44.1 or 48 kHz.
- **44.1 kHz** reduces USB bandwidth slightly and is equivalent for YIN-based
  pitch detection. Switch to 44.1 kHz only if your interface has known
  instability at 48 kHz, or to match a 44.1 kHz recording setup.

### Buffer size and hop size

The hop size controls how often the pitch detector runs. In the three latency
presets:

| Preset  | buf_size | hop_size | Detector runs every |
| ------- | -------- | -------- | ------------------- |
| low     | 1024     | 256      | ~5.3 ms at 48 kHz   |
| medium  | 2048     | 512      | ~10.7 ms at 48 kHz  |
| high    | 4096     | 1024     | ~21.3 ms at 48 kHz  |

On portable profile, `buf_size` is the ring buffer for YIN (not a DSP window),
so it can be increased without adding latency. On high_accuracy profile,
`buf_size` is the ring buffer; the spectral FFT window is fixed at 16384
samples (~341 ms at 48 kHz).

### ASIO buffer size override

When `asio_enabled = True` and `asio_buffer_size > 0`, the configured
`asio_buffer_size` value replaces the hop size as the PortAudio `blocksize`
parameter. Set this to match your ASIO driver's known-good buffer size when
the default hop size causes glitches.

### Testing procedure

Measurements above were taken with:

1. A loopback cable (mono TS 1/4" from headphone out to instrument input) for
   round-trip measurement.
2. Acoustic impulse at ~30 cm from the instrument input with the interface's
   gain at unity (12 o'clock).
3. The `measure_roundtrip_latency()` function in
   `pickhero/audio/latency_calibrator.py` using cross-correlation.

All values are approximate and depend on your specific system load, USB
controller, and driver version.
