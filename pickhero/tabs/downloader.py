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
    try:
        data = _urlopen(url)
    except (urllib.error.URLError, OSError):
        return []

    try:
        items = json.loads(data)
    except (json.JSONDecodeError, ValueError):
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


def _extract_source_from_state_script(page_html: str) -> str | None:
    """Extract the GP file CloudFront URL from the <script id="state"> JSON blob.

    Songsterr embeds tab metadata in a script tag like:
        <script id="state" type="application/json">{"meta": ...}</script>

    The source URL lives at state["meta"]["current"]["source"].

    Returns None if the URL can't be found.
    """
    match = re.search(
        r'<script\s+id=["\']state["\'][^>]*>(.*?)</script>',
        page_html,
        re.DOTALL,
    )
    if not match:
        return None

    try:
        state = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None

    try:
        source = state["meta"]["current"]["source"]
    except (KeyError, TypeError):
        return None

    if isinstance(source, str) and source.startswith("http"):
        return source
    return None


def get_songsterr_url(song_id: int) -> str:
    """Return the Songsterr browser URL for a given song ID."""
    return f"{SONGSTERR_TAB_URL}/{song_id}"


def download_gp5(song_id: int, output_path: str | Path) -> bool:
    """Download a GP5 tab file from Songsterr.

    Args:
        song_id: Songsterr song ID.
        output_path: Where to save the downloaded file.

    Returns:
        True if download succeeded, False otherwise.
    """
    tab_url = get_songsterr_url(song_id)
    try:
        page_data = _urlopen(tab_url)
        page_html = page_data.decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return False

    source_url = _extract_source_from_state_script(page_html)
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
