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
    device_index: int | None = None  # resolved at runtime
    device_name: str = ""  # preferred device name (resolved to index at runtime)
    sample_rate: int = 44100
    buf_size: int = 2048
    hop_size: int = 512
    confidence_threshold: float = 0.8
    onset_threshold: float = 0.3
    noise_gate_db: float = -60.0  # ignore signals below this dB level
    latency_mode: str = "medium"  # "low", "medium", "high"


# Latency presets: (buf_size, hop_size, description)
LATENCY_PRESETS = {
    "low": (1024, 256, "~12ms (may miss low notes)"),
    "medium": (2048, 512, "~23ms (balanced)"),
    "high": (4096, 1024, "~46ms (best detection)"),
}


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
    active_strings: list[bool] = field(default_factory=lambda: [True] * 8)
    chord_partial_credit: bool = True
    wait_mode: bool = False
    timing_judge_mode: bool = False
    pitch_strict_mode: bool = False
    sort_mode: str = "name_asc"
    calibration: dict = field(default_factory=dict)

    # Store default for HUD comparison (not serialized)
    _default_chord_partial_credit: bool = field(default=True, repr=False)

    def get_string_calibration(self, string: int) -> StringCalibration | None:
        """Return calibration for a string (1-N), or None if not calibrated."""
        strings = self.calibration.get("strings", {})
        data = strings.get(str(string))
        if data is None:
            return None
        return StringCalibration(**data)

    def set_string_calibration(self, string: int, cal: StringCalibration) -> None:
        """Store calibration for a string (1-N)."""
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
        """Load settings from JSON file.

        On a corrupt file (not JSON at all), returns defaults. On a file with
        unknown or mistyped fields, preserves the valid fields and drops only
        the bad ones — calibration data and audio device selection survive a
        single malformed entry instead of being wiped to defaults.
        """
        if not CONFIG_FILE.exists():
            return cls()
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # File is unreadable — nothing to preserve.
            return cls()

        # Pull nested config objects field-by-field so a single bad value
        # doesn't invalidate the whole file.
        audio = cls._filter_known(AudioConfig, data.pop("audio", {}))
        display = cls._filter_known(DisplayConfig, data.pop("display", {}))
        top = cls._filter_known(cls, data)
        return cls(audio=AudioConfig(**audio), display=DisplayConfig(**display), **top)

    @staticmethod
    def _filter_known(datacls: type, raw: dict) -> dict:
        """Return only the keys from ``raw`` that match ``datacls`` field names.

        Unknown keys (typos, fields from a newer PickHero version) are silently
        dropped; known keys are passed through so dataclass conversion can
        raise a clear TypeError if a value's type is wrong — caller decides.
        """
        if not isinstance(raw, dict) or not raw:
            return {}
        known = {f.name for f in datacls.__dataclass_fields__.values()}
        return {k: v for k, v in raw.items() if k in known}
