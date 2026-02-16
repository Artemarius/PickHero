"""User settings management.

Settings stored as JSON in the user's home directory.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_DIR = Path.home() / ".pickhero"
CONFIG_FILE = CONFIG_DIR / "settings.json"


@dataclass
class StringCalibration:
    """Calibration data for a single guitar string."""
    midi_note: int        # detected MIDI note (e.g. 40 for E2)
    frequency: float      # median detected frequency (Hz)
    noise_floor_db: float  # noise floor measured before playing


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
    count_in_beats: int = 4
    theme: str = "dark"
    max_fret: int = 24
    active_strings: list[bool] = field(default_factory=lambda: [True] * 6)
    chord_partial_credit: bool = True
    sort_mode: str = "name_asc"
    calibration: dict = field(default_factory=dict)

    # Store default for HUD comparison (not serialized)
    _default_chord_partial_credit: bool = field(default=True, repr=False)

    def get_string_calibration(self, string: int) -> StringCalibration | None:
        """Return calibration for a string (1-6), or None if not calibrated."""
        strings = self.calibration.get("strings", {})
        data = strings.get(str(string))
        if data is None:
            return None
        return StringCalibration(**data)

    def set_string_calibration(self, string: int, cal: StringCalibration) -> None:
        """Store calibration for a string (1-6)."""
        if "strings" not in self.calibration:
            self.calibration["strings"] = {}
        self.calibration["strings"][str(string)] = asdict(cal)

    def is_calibrated(self) -> bool:
        """True if at least one string has been calibrated."""
        strings = self.calibration.get("strings", {})
        return len(strings) > 0

    def save(self):
        """Save settings to JSON file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("_default_chord_partial_credit", None)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> "Config":
        """Load settings from JSON file. Returns defaults if file doesn't exist."""
        if not CONFIG_FILE.exists():
            return cls()
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            data.pop("_default_chord_partial_credit", None)
            audio_data = data.pop("audio", {})
            display_data = data.pop("display", {})
            return cls(
                audio=AudioConfig(**audio_data),
                display=DisplayConfig(**display_data),
                **data,
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            return cls()
