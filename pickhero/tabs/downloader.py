"""Songsterr tab search and download.

Provides search and download functionality for Guitar Pro tabs from Songsterr.
Falls back gracefully when downloads require Songsterr Plus authentication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests

SONGSTERR_API_URL = "https://www.songsterr.com/a/ra/songs.json"
SONGSTERR_TAB_URL = "https://www.songsterr.com/a/wsa"

REQUEST_TIMEOUT = 15


@dataclass
class SongsterrResult:
    """A search result from Songsterr."""

    song_id: int
    title: str
    artist: str


def search(query: str, max_results: int = 10) -> list[SongsterrResult]:
    """Search Songsterr for tabs matching a query.

    Args:
        query: Search string (song name, artist, etc.).
        max_results: Maximum results to return.

    Returns:
        List of SongsterrResult, empty on error.
    """
    try:
        resp = requests.get(
            SONGSTERR_API_URL,
            params={"pattern": query},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []

    results = []
    for item in resp.json()[:max_results]:
        results.append(
            SongsterrResult(
                song_id=item.get("id", 0),
                title=item.get("title", ""),
                artist=item.get("artist", {}).get("name", ""),
            )
        )
    return results


def download_gp5(song_id: int, output_path: str | Path) -> bool:
    """Download a GP5 tab file from Songsterr.

    Args:
        song_id: Songsterr song ID.
        output_path: Where to save the downloaded file.

    Returns:
        True if download succeeded, False otherwise (e.g. auth required).
    """
    # Songsterr serves tab source files at a predictable URL pattern
    tab_url = f"{SONGSTERR_TAB_URL}/{song_id}"
    try:
        # Fetch the tab page to find the source GP file URL
        resp = requests.get(tab_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # Look for the GP file source URL in the page
        # Songsterr embeds it as a data attribute or in JSON config
        source_url = _extract_source_url(resp.text, song_id)
        if not source_url:
            return False

        # Download the actual GP file
        file_resp = requests.get(source_url, timeout=REQUEST_TIMEOUT)
        file_resp.raise_for_status()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(file_resp.content)
        return True

    except requests.RequestException:
        return False


def _extract_source_url(page_html: str, song_id: int) -> str | None:
    """Extract the GP file download URL from a Songsterr tab page.

    Returns None if the URL can't be found (e.g. requires Songsterr Plus).
    """
    # Try JSON-embedded source URL
    match = re.search(r'"source"\s*:\s*"(https?://[^"]+\.gp[345])"', page_html)
    if match:
        return match.group(1)

    # Try alternative pattern with revision data
    match = re.search(
        r'"source"\s*:\s*"(https?://[^"]+(?:guitar-pro|tab)[^"]*)"', page_html
    )
    if match:
        return match.group(1)

    return None
