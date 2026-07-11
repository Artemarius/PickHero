"""Tested device matrix for PickHero audio input.

Provides a ``DEVICE_MATRIX`` list of dicts documenting tested audio interfaces,
platforms, backends, and measured latencies, plus helper functions for
recommending and validating device configurations.

Each entry in ``DEVICE_MATRIX`` has this shape::

    {
        "device_name": str,          # display name used in docs/logs
        "platforms": list[str],      # "linux", "windows", "darwin"
        "backends": list[str],       # e.g. ["ALSA", "PulseAudio", "PipeWire"]
        "sample_rates": list[int],   # e.g. [44100, 48000]
        "buffer_sizes": list[int],   # e.g. [256, 512, 1024, 2048]
        "input_latency_ms": str,     # typical range as a short string
        "recommended": str,          # recommended latency preset name
        "notes": str,                # known issues or caveats
    }

The matching functions use substring matching so a device name reported
by ``sounddevice.query_devices()`` like ``"ASIO: Focusrite USB ASIO"``
will still match ``"Focusrite Scarlett Solo"``.

Constants
---------
DEVICE_MATRIX : list[dict]
    The full tested-device matrix.
LATENCY_PRESET_BUF_HOP : dict[str, tuple[int, int]]
    Maps latency mode to ``(buf_size, hop_size)`` — mirrors
    ``pickhero.config.LATENCY_PRESETS``.

Functions
---------
get_recommended_settings(device_name, platform)
    Return recommended sample_rate, buffer_size, hop_size, latency_mode
    for a given device and OS.
validate_device_config(device_name, platform, sample_rate, buffer_size)
    Return a list of warning strings for unsupported combinations.
"""

from __future__ import annotations

import sys
from typing import Any

# ---------------------------------------------------------------------------
# Device matrix
# ---------------------------------------------------------------------------

DEVICE_MATRIX: list[dict[str, Any]] = [
    {
        "device_name": "Focusrite Scarlett Solo (3rd gen)",
        "platforms": ["linux", "win32", "darwin"],
        "backends": ["ALSA", "PulseAudio", "PipeWire", "WASAPI", "ASIO", "CoreAudio"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [256, 512, 1024, 2048],
        "input_latency_ms": "4–14 ms",
        "recommended": "medium",
        "notes": (
            "Works well across all platforms. ASIO or WASAPI exclusive "
            "on Windows; CoreAudio on macOS; ALSA via PulseAudio/PipeWire on Linux. "
            "ASIO channel selector must point at input 1."
        ),
    },
    {
        "device_name": "Focusrite Scarlett 2i2 (3rd gen)",
        "platforms": ["linux", "win32", "darwin"],
        "backends": ["ALSA", "PulseAudio", "PipeWire", "WASAPI", "ASIO", "CoreAudio"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [256, 512, 1024, 2048],
        "input_latency_ms": "4–14 ms",
        "recommended": "medium",
        "notes": (
            "Two inputs; set input_channel to 0 (left) or 1 (right). "
            "Power-on pop may briefly saturate the input."
        ),
    },
    {
        "device_name": "Behringer UMC22",
        "platforms": ["linux", "win32", "darwin"],
        "backends": ["ALSA", "PulseAudio", "WASAPI", "ASIO4ALL", "CoreAudio"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [512, 1024, 2048],
        "input_latency_ms": "10–20 ms",
        "recommended": "high",
        "notes": (
            "256 buffer is unstable across all platforms. Use at least 512. "
            "On Windows ASIO4ALL buffer below 256 causes crackling; "
            "latency_mode='low' may fail with ASIO4ALL. "
            "On macOS kernel extension (kext) may be needed on Intel machines."
        ),
    },
    {
        "device_name": "Behringer UMC202HD",
        "platforms": ["linux", "win32", "darwin"],
        "backends": ["ALSA", "PulseAudio", "WASAPI", "ASIO", "CoreAudio"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [512, 1024, 2048],
        "input_latency_ms": "8–18 ms",
        "recommended": "medium",
        "notes": (
            "Native ASIO driver on Windows only supports 44.1 kHz at 256 buffer minimum. "
            "On older Linux kernels (6.1–6.5) may need snd-usb-audio quirks. "
            "macOS may need USB audio class compliance mode."
        ),
    },
    {
        "device_name": "Built-in audio",
        "platforms": ["linux", "win32", "darwin"],
        "backends": ["ALSA", "PulseAudio", "PipeWire", "WASAPI", "CoreAudio"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [512, 1024, 2048],
        "input_latency_ms": "10–40 ms",
        "recommended": "high",
        "notes": (
            "Higher jitter than USB interfaces. On Linux use PipeWire or "
            "PulseAudio; on Windows use WASAPI shared (exclusive may fail "
            "with many Realtek codecs); on macOS the 3.5 mm jack has higher "
            "noise floor. A USB interface is recommended for reliable detection."
        ),
    },
    {
        "device_name": "ASIO4ALL (generic WDM wrapper)",
        "platforms": ["win32"],
        "backends": ["ASIO4ALL"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [512, 1024, 2048],
        "input_latency_ms": "12–25 ms",
        "recommended": "medium",
        "notes": (
            "ASIO4ALL wraps WDM drivers; reliability varies per underlying "
            "device. Buffer size below 512 may glitch. Use native ASIO or "
            "WASAPI exclusive when available."
        ),
    },
    {
        "device_name": "Built-in Realtek (ALC892/ALC1220)",
        "platforms": ["win32"],
        "backends": ["WASAPI"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [1024, 2048],
        "input_latency_ms": "20–40 ms",
        "recommended": "high",
        "notes": (
            "WASAPI exclusive may not work on all Realtek codecs — falls back "
            "to WASAPI shared automatically. High latency makes PickHero "
            "playable but detection accuracy suffers."
        ),
    },
    {
        "device_name": "Built-in (MacBook Pro)",
        "platforms": ["darwin"],
        "backends": ["CoreAudio"],
        "sample_rates": [44100, 48000],
        "buffer_sizes": [512, 1024, 2048],
        "input_latency_ms": "10–20 ms",
        "recommended": "medium",
        "notes": (
            "3.5 mm jack has higher noise floor than USB interfaces. "
            "Microphone permission must be granted in System Preferences > "
            "Security & Privacy > Microphone."
        ),
    },
]

# Windows device-name prefixes to strip for matching
_WIN_BACKEND_PREFIXES = ("ASIO: ", "WASAPI: ", "MME: ", "DirectSound: ")

# Map from latency preset to (buf_size, hop_size) — mirrors config.LATENCY_PRESETS
LATENCY_PRESET_BUF_HOP: dict[str, tuple[int, int]] = {
    "low": (1024, 256),
    "medium": (2048, 512),
    "high": (4096, 1024),
}

_MIN_BUF_BY_PRESET: dict[str, int] = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_backend_prefix(name: str) -> str:
    """Remove known PortAudio backend prefixes from a device name."""
    n = name.strip()
    for prefix in _WIN_BACKEND_PREFIXES:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return n


def _normalize_name(name: str) -> str:
    """Strip backend prefixes and lower-case for fuzzy matching."""
    return _strip_backend_prefix(name).lower()


def _significant_words(name: str) -> set[str]:
    """Return words >= 4 chars from a name (stopwords excluded)."""
    _stopwords = {"builtin", "built-in", "generic", "wrapper", "wdm",
                  "gen", "nd", "rd", "th", "alc"}
    words = set()
    for w in name.replace("(", "").replace(")", "").replace("-", " ").split():
        w = w.strip(",. ")
        if len(w) >= 4 and w.lower() not in _stopwords:
            words.add(w.lower())
    return words


def _match_device(device_name: str, platform: str | None = None) -> dict[str, Any] | None:
    """Find the best-matching entry in DEVICE_MATRIX.

    Matching strategy (in order):
    1. Exact substring match (case-insensitive, backend-prefix-stripped).
    2. Word overlap — both names share at least two significant words
       (>= 4 chars, excluding stopwords like ``builtin``, ``generic``).
    3. Single-word brand match — fallback when the caller provides
       an unfamiliar variant that shares at least one significant word.
    Returns the best-scoring match or None.
    """
    stripped = _strip_backend_prefix(device_name)
    needle = stripped.lower()
    needle_words = _significant_words(stripped)

    candidates: list[tuple[int, int, dict[str, Any]]] = []

    for entry in DEVICE_MATRIX:
        if platform is not None and platform not in entry["platforms"]:
            continue

        entry_name_raw = entry["device_name"]
        entry_name = entry_name_raw.lower()
        entry_words = _significant_words(entry_name_raw)

        # Strategy 1: exact substring match
        if entry_name in needle or needle in entry_name:
            # Score by length: longer name = more specific match
            candidates.append((1, len(entry_name), entry))
            continue

        # Strategy 2: word overlap (at least 2 significant words in common)
        overlap = needle_words & entry_words
        if len(overlap) >= 2:
            candidates.append((0, len(overlap), entry))
            continue

        # Strategy 3: single-word brand match
        common = needle_words & entry_words
        if len(common) >= 1 and len(needle_words) >= 2 and len(entry_words) >= 2:
            candidates.append((-1, len(entry_name), entry))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[:2], reverse=True)
    return candidates[0][2]


def get_recommended_settings(
    device_name: str,
    platform: str | None = None,
    profile: str = "portable",
) -> dict[str, Any]:
    """Return recommended audio config for *device_name* on *platform*.

    Parameters
    ----------
    device_name : str
        Device name as printed by ``sounddevice.query_devices()`` or
        ``AudioCapture.list_audio_devices()``.
    platform : str or None
        OS platform string (``sys.platform`` style). When ``None``, the
        current platform is used.
    profile : str
        Detection profile — ``"portable"`` (default) or ``"high_accuracy"``.

    Returns
    -------
    dict with keys ``sample_rate``, ``buffer_size``, ``hop_size``,
    ``latency_mode``, and ``asio_enabled``.

    If no device match is found, returns portable defaults.
    """
    if platform is None:
        platform = sys.platform

    entry = _match_device(device_name, platform)

    if entry is None:
        # Fallback: safe portable defaults
        return {
            "sample_rate": 44100,
            "buffer_size": 2048,
            "hop_size": 512,
            "latency_mode": "medium",
            "asio_enabled": False,
        }

    rec_preset = entry.get("recommended", "medium")
    buf, hop = LATENCY_PRESET_BUF_HOP.get(rec_preset, (2048, 512))
    sample_rate = 48000 if profile == "high_accuracy" else 48000

    asio = platform == "win32" and any(
        a in (b or "").upper() for a in ("ASIO",) for b in entry.get("backends", [])
    )

    return {
        "sample_rate": sample_rate,
        "buffer_size": buf,
        "hop_size": hop,
        "latency_mode": rec_preset,
        "asio_enabled": asio,
    }


def validate_device_config(
    device_name: str,
    platform: str | None = None,
    sample_rate: int = 48000,
    buffer_size: int = 2048,
) -> list[str]:
    """Return a list of warning strings for unsupported config combinations.

    Parameters
    ----------
    device_name : str
        Device name as printed by ``sounddevice.query_devices()``.
    platform : str or None
        OS platform string (``sys.platform`` style). When ``None``, the
        current platform is used.
    sample_rate : int
        Requested sample rate (44100 or 48000 typically).
    buffer_size : int
        Requested buffer size in frames.

    Returns
    -------
    list of str
        Warning messages. Empty list means the combination is fully supported.
    """
    if platform is None:
        platform = sys.platform

    warnings: list[str] = []
    entry = _match_device(device_name, platform)

    if entry is None:
        return []  # unknown device — cannot validate

    # Check sample rate
    if sample_rate not in entry.setdefault("sample_rates", [44100, 48000]):
        supported = entry["sample_rates"]
        warnings.append(
            f"Sample rate {sample_rate} is not in the tested list for "
            f"{entry['device_name']} on {platform}. Tested: {supported}. "
            f"Use one of {supported} for best results."
        )

    # Check buffer size
    if buffer_size not in entry.setdefault("buffer_sizes", [256, 512, 1024, 2048]):
        supported = entry["buffer_sizes"]
        warnings.append(
            f"Buffer size {buffer_size} is not in the tested list for "
            f"{entry['device_name']} on {platform}. Tested: {supported}. "
            f"Use one of {supported} for best results."
        )

    # Buffer too small for preset
    if buffer_size < 256:
        warnings.append(
            f"Buffer size {buffer_size} is below the minimum supported size "
            f"(256). Expect glitches or stream failure."
        )

    if buffer_size < 512 and entry.get("recommended") == "high":
        warnings.append(
            f"Buffer size {buffer_size} is below the recommended minimum "
            f"for {entry['device_name']} (at least 512). "
            f"Expect audio glitches."
        )

    return warnings


# ---------------------------------------------------------------------------
# Convenience shortcuts
# ---------------------------------------------------------------------------

MIN_BUF_BY_PRESET = _MIN_BUF_BY_PRESET
"""dict[str, int]: Minimum buffer size recommended per latency preset."""
