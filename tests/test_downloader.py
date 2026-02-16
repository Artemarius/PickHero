"""Tests for pickhero.tabs.downloader module."""

import json
from unittest.mock import patch

from pickhero.tabs.downloader import (
    SongsterrResult,
    _extract_source_from_state_script,
    download_gp5,
    get_songsterr_url,
    sanitize_filename,
    search,
)


class TestSearch:
    def test_returns_results(self):
        api_response = json.dumps([
            {"songId": 123, "title": "Smoke on the Water", "artist": "Deep Purple"},
            {"songId": 456, "title": "Paranoid", "artist": "Black Sabbath"},
        ]).encode()

        with patch("pickhero.tabs.downloader._urlopen", return_value=api_response):
            results = search("smoke")

        assert len(results) == 2
        assert results[0] == SongsterrResult(123, "Smoke on the Water", "Deep Purple")
        assert results[1] == SongsterrResult(456, "Paranoid", "Black Sabbath")

    def test_max_results(self):
        items = [{"songId": i, "title": f"Song {i}", "artist": "A"} for i in range(20)]
        api_response = json.dumps(items).encode()

        with patch("pickhero.tabs.downloader._urlopen", return_value=api_response):
            results = search("test", max_results=5)

        assert len(results) == 5

    def test_network_error_returns_empty(self):
        import urllib.error
        with patch(
            "pickhero.tabs.downloader._urlopen",
            side_effect=urllib.error.URLError("fail"),
        ):
            results = search("anything")
        assert results == []

    def test_malformed_json_returns_empty(self):
        with patch("pickhero.tabs.downloader._urlopen", return_value=b"not json"):
            results = search("anything")
        assert results == []

    def test_missing_fields_use_defaults(self):
        api_response = json.dumps([{"other": "data"}]).encode()
        with patch("pickhero.tabs.downloader._urlopen", return_value=api_response):
            results = search("test")

        assert len(results) == 1
        assert results[0].song_id == 0
        assert results[0].title == ""
        assert results[0].artist == ""


class TestExtractSourceFromStateScript:
    def test_extracts_url(self):
        source_url = "https://d12345.cloudfront.net/tabs/song.gp5"
        state = {"meta": {"current": {"source": source_url}}}
        html = f'<script id="state" type="application/json">{json.dumps(state)}</script>'

        result = _extract_source_from_state_script(html)
        assert result == source_url

    def test_double_quotes_around_id(self):
        state = {"meta": {"current": {"source": "https://cdn.example.com/tab.gp5"}}}
        html = f'<script id="state">{json.dumps(state)}</script>'
        assert _extract_source_from_state_script(html) == "https://cdn.example.com/tab.gp5"

    def test_single_quotes_around_id(self):
        state = {"meta": {"current": {"source": "https://cdn.example.com/tab.gp5"}}}
        html = f"<script id='state'>{json.dumps(state)}</script>"
        assert _extract_source_from_state_script(html) == "https://cdn.example.com/tab.gp5"

    def test_no_script_tag(self):
        assert _extract_source_from_state_script("<html><body>hi</body></html>") is None

    def test_invalid_json(self):
        html = '<script id="state">not valid json</script>'
        assert _extract_source_from_state_script(html) is None

    def test_missing_meta_key(self):
        html = f'<script id="state">{json.dumps({"other": "data"})}</script>'
        assert _extract_source_from_state_script(html) is None

    def test_missing_current_key(self):
        html = f'<script id="state">{json.dumps({"meta": {"other": 1}})}</script>'
        assert _extract_source_from_state_script(html) is None

    def test_missing_source_key(self):
        state = {"meta": {"current": {"revision": 1}}}
        html = f'<script id="state">{json.dumps(state)}</script>'
        assert _extract_source_from_state_script(html) is None

    def test_non_http_source_rejected(self):
        state = {"meta": {"current": {"source": "ftp://example.com/tab.gp5"}}}
        html = f'<script id="state">{json.dumps(state)}</script>'
        assert _extract_source_from_state_script(html) is None

    def test_non_string_source_rejected(self):
        state = {"meta": {"current": {"source": 12345}}}
        html = f'<script id="state">{json.dumps(state)}</script>'
        assert _extract_source_from_state_script(html) is None


class TestDownloadGp5:
    def test_success(self, tmp_path):
        source_url = "https://cdn.example.com/tab.gp5"
        state = {"meta": {"current": {"source": source_url}}}
        page_html = f'<script id="state">{json.dumps(state)}</script>'
        file_bytes = b"\x00GP5_FAKE_DATA"

        def fake_urlopen(url, timeout=15):
            if "wsa" in url:
                return page_html.encode()
            return file_bytes

        output = tmp_path / "test.gp5"
        with patch("pickhero.tabs.downloader._urlopen", side_effect=fake_urlopen):
            result = download_gp5(42, output)

        assert result is True
        assert output.read_bytes() == file_bytes

    def test_page_fetch_fails(self, tmp_path):
        import urllib.error
        with patch(
            "pickhero.tabs.downloader._urlopen",
            side_effect=urllib.error.URLError("fail"),
        ):
            result = download_gp5(42, tmp_path / "test.gp5")
        assert result is False

    def test_no_source_url_in_page(self, tmp_path):
        page_html = b"<html><body>no state script</body></html>"
        with patch("pickhero.tabs.downloader._urlopen", return_value=page_html):
            result = download_gp5(42, tmp_path / "test.gp5")
        assert result is False

    def test_file_download_fails(self, tmp_path):
        import urllib.error
        source_url = "https://cdn.example.com/tab.gp5"
        state = {"meta": {"current": {"source": source_url}}}
        page_html = f'<script id="state">{json.dumps(state)}</script>'

        call_count = 0

        def fake_urlopen(url, timeout=15):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page_html.encode()
            raise urllib.error.URLError("download fail")

        output = tmp_path / "test.gp5"
        with patch("pickhero.tabs.downloader._urlopen", side_effect=fake_urlopen):
            result = download_gp5(42, output)

        assert result is False
        assert not output.exists()

    def test_creates_parent_dirs(self, tmp_path):
        source_url = "https://cdn.example.com/tab.gp5"
        state = {"meta": {"current": {"source": source_url}}}
        page_html = f'<script id="state">{json.dumps(state)}</script>'

        def fake_urlopen(url, timeout=15):
            if "wsa" in url:
                return page_html.encode()
            return b"data"

        output = tmp_path / "sub" / "dir" / "test.gp5"
        with patch("pickhero.tabs.downloader._urlopen", side_effect=fake_urlopen):
            result = download_gp5(42, output)

        assert result is True
        assert output.exists()


class TestGetSongsterrUrl:
    def test_format(self):
        assert get_songsterr_url(123) == "https://www.songsterr.com/a/wsa/123"

    def test_different_id(self):
        assert get_songsterr_url(999) == "https://www.songsterr.com/a/wsa/999"


class TestSanitizeFilename:
    def test_removes_invalid_chars(self):
        assert sanitize_filename('AC/DC - Back In Black') == "ACDC - Back In Black"

    def test_removes_multiple_chars(self):
        assert sanitize_filename('test<>:"/\\|?*end') == "testend"

    def test_strips_whitespace(self):
        assert sanitize_filename("  hello  ") == "hello"

    def test_leaves_valid_chars(self):
        assert sanitize_filename("Artist - Song (Live)") == "Artist - Song (Live)"
