"""User settings management.

Settings stored as JSON in the user's home directory.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_DIR = Path.home() / ".pickhero"
CONFIG_FILE = CONFIG_DIR / "settings.json"


@dataclass
class AudioConfig:
    """Audio capture and detection settings."""
    device_index: int | None = None  # None = system default
    sample_rate: int = 44100
    buf_size: int = 2048
    hop_size: int = 512
    confidence_threshold: float = 0.8
    onset_threshold: float = 0.3
    noise_gate_db: float = -60.0  # ignore signals below this dB level


@dataclass
class DisplayConfig:
    """Display and rendering settings."""
    width: int = 1280
    height: int = 720
    visible_beats: int = 16
    hit_zone_fraction: float = 0.20


@dataclass
class Config:
    """Application settings."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    songs_dir: str = "songs"
    tempo_factor: float = 1.0
    timing_window_ms: float = 100.0
    audio_latency_offset_ms: float = 0.0
    chord_threshold_ms: float = 50.0
    backing_track_enabled: bool = True

    def save(self):
        """Save settings to JSON file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "Config":
        """Load settings from JSON file. Returns defaults if file doesn't exist."""
        if not CONFIG_FILE.exists():
            return cls()
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            audio_data = data.pop("audio", {})
            display_data = data.pop("display", {})
            return cls(
                audio=AudioConfig(**audio_data),
                display=DisplayConfig(**display_data),
                **data,
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return cls()
