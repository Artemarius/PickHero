"""Base importer interface for guitar datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pickhero.datasets.schema import ClipEvent


class DatasetImporter(ABC):
    """Importer from a dataset directory to a list of ClipEvent records."""

    name: str = ""

    @abstractmethod
    def scan(self, path: str | Path) -> list[ClipEvent]:
        """Walk ``path`` and yield normalized ClipEvent records."""
        ...

    def _audio_files(self, path: Path) -> list[Path]:
        """Return all WAV/FLAC files under ``path``."""
        return sorted(path.rglob("*.wav")) + sorted(path.rglob("*.flac"))
