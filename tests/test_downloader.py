"""Tests for pickhero.tabs.downloader module."""

import json
import urllib.error
from unittest.mock import patch

from pickhero.tabs.downloader import (
    SongsterrResult,
    _get_source_url,
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


class TestGetSourceUrl:
    def _make_responses(self, meta_body, revision_body=None):
        """Helper: return a fake _urlopen that returns meta then revision JSON."""
        responses = iter(
            [json.dumps(b).encode() for b in [meta_body] + ([revision_body] if revision_body else [])]
        )
        def fake_urlopen(url, timeout=15):
            try:
                return next(responses)
            except StopIteration:
                raise urllib.error.URLError("unexpected call")
        return fake_urlopen

    def test_success(self):
        meta = {"revisionId": 999}
        revision = {"source": "https://gp.songsterr.com/export.abc.gp"}
        with patch("pickhero.tabs.downloader._urlopen",
                   side_effect=self._make_responses(meta, revision)):
            result = _get_source_url(42)
        assert result == "https://gp.songsterr.com/export.abc.gp"

    def test_meta_network_error(self):
        with patch("pickhero.tabs.downloader._urlopen",
                   side_effect=urllib.error.URLError("fail")):
            assert _get_source_url(42) is None

    def test_meta_not_dict(self):
        with patch("pickhero.tabs.downloader._urlopen",
                   return_value=json.dumps([1, 2]).encode()):
            assert _get_source_url(42) is None

    def test_meta_missing_revision_id(self):
        with patch("pickhero.tabs.downloader._urlopen",
                   return_value=json.dumps({"other": "data"}).encode()):
            assert _get_source_url(42) is None

    def test_revision_network_error(self):
        meta = {"revisionId": 999}
        calls = [0]
        def fake(url, timeout=15):
            calls[0] += 1
            if calls[0] == 1:
                return json.dumps(meta).encode()
            raise urllib.error.URLError("fail")
        with patch("pickhero.tabs.downloader._urlopen", side_effect=fake):
            assert _get_source_url(42) is None

    def test_revision_missing_source(self):
        meta = {"revisionId": 999}
        revision = {"other": "data"}
        with patch("pickhero.tabs.downloader._urlopen",
                   side_effect=self._make_responses(meta, revision)):
            assert _get_source_url(42) is None

    def test_non_http_source_rejected(self):
        meta = {"revisionId": 999}
        revision = {"source": "ftp://example.com/tab.gp5"}
        with patch("pickhero.tabs.downloader._urlopen",
                   side_effect=self._make_responses(meta, revision)):
            assert _get_source_url(42) is None

    def test_non_string_source_rejected(self):
        meta = {"revisionId": 999}
        revision = {"source": 12345}
        with patch("pickhero.tabs.downloader._urlopen",
                   side_effect=self._make_responses(meta, revision)):
            assert _get_source_url(42) is None


class TestDownloadGp5:
    def test_success(self, tmp_path):
        source_url = "https://gp.songsterr.com/export.abc.gp"
        file_bytes = b"\x00GP5_FAKE_DATA"

        def fake_urlopen(url, timeout=15):
            if "meta" in url:
                return json.dumps({"revisionId": 999}).encode()
            if "revision" in url:
                return json.dumps({"source": source_url}).encode()
            return file_bytes

        output = tmp_path / "test.gp5"
        with patch("pickhero.tabs.downloader._urlopen", side_effect=fake_urlopen):
            result = download_gp5(42, output)

        assert result is True
        assert output.read_bytes() == file_bytes

    def test_source_url_not_found(self, tmp_path):
        with patch("pickhero.tabs.downloader._urlopen",
                   side_effect=urllib.error.URLError("fail")):
            result = download_gp5(42, tmp_path / "test.gp5")
        assert result is False

    def test_file_download_fails(self, tmp_path):
        source_url = "https://gp.songsterr.com/export.abc.gp"
        calls = [0]

        def fake_urlopen(url, timeout=15):
            calls[0] += 1
            if calls[0] == 1:
                return json.dumps({"revisionId": 999}).encode()
            if calls[0] == 2:
                return json.dumps({"source": source_url}).encode()
            raise urllib.error.URLError("download fail")

        output = tmp_path / "test.gp5"
        with patch("pickhero.tabs.downloader._urlopen", side_effect=fake_urlopen):
            result = download_gp5(42, output)

        assert result is False
        assert not output.exists()

    def test_creates_parent_dirs(self, tmp_path):
        source_url = "https://gp.songsterr.com/export.abc.gp"

        def fake_urlopen(url, timeout=15):
            if "meta" in url:
                return json.dumps({"revisionId": 999}).encode()
            if "revision" in url:
                return json.dumps({"source": source_url}).encode()
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
