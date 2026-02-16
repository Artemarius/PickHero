"""Songsterr tab search and download.

Provides search and download functionality for Guitar Pro tabs from Songsterr.
Falls back to opening the browser when direct download fails.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SONGSTERR_API_URL = "https://www.songsterr.com/api/songs"
SONGSTERR_META_URL = "https://www.songsterr.com/api/meta"
SONGSTERR_REVISION_URL = "https://www.songsterr.com/api/revision"
SONGSTERR_TAB_URL = "https://www.songsterr.com/a/wsa"

REQUEST_TIMEOUT = 15

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class SongsterrResult:
    """A search result from Songsterr."""

    song_id: int
    title: str
    artist: str


def _urlopen(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    """Fetch a URL with browser-like headers. Returns response body bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_json(url: str) -> dict | list | None:
    """Fetch a URL and parse as JSON. Returns None on any error."""
    try:
        data = _urlopen(url)
    except (urllib.error.URLError, OSError):
        return None
    try:
        return json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return None


def search(query: str, max_results: int = 10) -> list[SongsterrResult]:
    """Search Songsterr for tabs matching a query.

    Args:
        query: Search string (song name, artist, etc.).
        max_results: Maximum results to return.

    Returns:
        List of SongsterrResult, empty on error.
    """
    params = urllib.parse.urlencode({"pattern": query})
    url = f"{SONGSTERR_API_URL}?{params}"
    items = _fetch_json(url)
    if not isinstance(items, list):
        return []

    results = []
    for item in items[:max_results]:
        results.append(
            SongsterrResult(
                song_id=item.get("songId", 0),
                title=item.get("title", ""),
                artist=item.get("artist", ""),
            )
        )
    return results


def _get_source_url(song_id: int) -> str | None:
    """Get the GP file download URL via Songsterr's API.

    Two API calls:
        1. /api/meta/{songId} → revisionId
        2. /api/revision/{revisionId} → source URL

    Returns None if the source URL can't be resolved.
    """
    meta = _fetch_json(f"{SONGSTERR_META_URL}/{song_id}")
    if not isinstance(meta, dict):
        return None

    revision_id = meta.get("revisionId")
    if not revision_id:
        return None

    revision = _fetch_json(f"{SONGSTERR_REVISION_URL}/{revision_id}")
    if not isinstance(revision, dict):
        return None

    source = revision.get("source")
    if isinstance(source, str) and source.startswith("http"):
        return source
    return None


def get_songsterr_url(song_id: int) -> str:
    """Return the Songsterr browser URL for a given song ID."""
    return f"{SONGSTERR_TAB_URL}/{song_id}"


def download_gp5(song_id: int, output_path: str | Path) -> bool:
    """Download a GP tab file from Songsterr.

    Args:
        song_id: Songsterr song ID.
        output_path: Where to save the downloaded file.

    Returns:
        True if download succeeded, False otherwise.
    """
    source_url = _get_source_url(song_id)
    if not source_url:
        return False

    try:
        file_data = _urlopen(source_url)
    except (urllib.error.URLError, OSError):
        return False

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(file_data)
    return True


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in filenames."""
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()
