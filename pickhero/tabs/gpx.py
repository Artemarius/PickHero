"""Guitar Pro 6 (.gpx) container decompression.

A ``.gpx`` file is a two-layer container holding ``score.gpif`` XML — the
same XML schema GP7/GP8 uses, so after decompression the existing GP7 parser
can read it.

Layers:
    1. **BCFZ** — outer wrapper. 4-byte ``BCFZ`` magic, a little-endian int32
       giving the decompressed size, then a custom bit-level LZ77 bitstream.
    2. **BCFS** — the decompressed BCFZ payload is a sector-based filesystem
       (0x1000-byte sectors) whose directory entries point to the file data.

The compression is **not** zlib/deflate; it is a bespoke bit-level LZ77 with
no entropy coding. Only :mod:`struct` is needed.

Format reference: PyGuitarPro ``gpx.py`` (J. Jørgen von Bargen), TuxGuitar
``GPXFileSystem.java``, alphaTab ``GpxFileSystem.ts`` — all four independent
implementations agree on the layout.
"""

from __future__ import annotations

import struct
from pathlib import Path

_BCFS = b"BCFS"
_BCFZ = b"BCFZ"
_SECTOR = 0x1000


class _BitReader:
    """MSB-first bit reader over a bytes buffer."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read_bit(self) -> int:
        byte_index = self._pos >> 3
        if byte_index >= len(self._data):
            return 0
        bit_index = 7 - (self._pos & 7)
        self._pos += 1
        return (self._data[byte_index] >> bit_index) & 1

    def read_bits(self, count: int, *, reversed_: bool = False) -> int:
        """Read ``count`` bits. If ``reversed_``, the LSB-first bit order used
        by the GPX offset/length fields is applied."""
        if count <= 0:
            return 0
        bits = [self.read_bit() for _ in range(count)]
        if reversed_:
            bits = list(reversed(bits))
        value = 0
        for b in bits:
            value = (value << 1) | b
        return value

    def read_bytes(self, n: int) -> bytes:
        # Bytes are read MSB-first via the bit stream to stay aligned.
        out = bytearray(n)
        for i in range(n):
            out[i] = self.read_bits(8)
        return bytes(out)


def decompress_bcfz(data: bytes) -> bytes:
    """Decompress a BCFZ-compressed ``.gpx`` file body into the BCFS payload.

    ``data`` is the file contents *after* the 4-byte ``BCFZ`` magic has been
    consumed (or the whole file — the magic is checked here too for safety).
    Returns the raw BCFS filesystem image.
    """
    # Tolerate being handed the whole file or the body after the magic.
    if data[:4] == _BCFZ:
        data = data[4:]
    elif data[:4] == _BCFS:
        # Already decompressed (BCFS passed through) — return as-is.
        return data

    expected, = struct.unpack_from("<i", data, 0)
    if expected <= 0:
        return b""

    reader = _BitReader(data[4:])
    out = bytearray()
    while len(out) < expected:
        flag = reader.read_bit()
        if flag:
            # Back-reference: copy from earlier in the output.
            word_size = reader.read_bits(4)
            offset = reader.read_bits(word_size, reversed_=True)
            size = reader.read_bits(word_size, reversed_=True)
            if offset == 0:
                # Degenerate token — stop to avoid an infinite zero-copy loop.
                break
            start = len(out) - offset
            if start < 0:
                # Malformed stream — can't reference before the start.
                break
            to_read = min(offset, size)
            out += out[start:start + to_read]
        else:
            # Literal run: 0-3 raw bytes.
            size = reader.read_bits(2, reversed_=True)
            out += reader.read_bytes(size)
    return bytes(out)


def _read_int(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _read_filename(data: bytes, offset: int, length: int = 127) -> str:
    raw = data[offset:offset + length]
    raw = raw.split(b"\x00", 1)[0]
    return raw.decode("cp1252", errors="replace")


def parse_bcfs(data: bytes) -> dict[str, bytes]:
    """Parse a BCFS filesystem image into ``{filename: bytes}``.

    ``data`` is the decompressed BCFZ payload (or a raw BCFS image). The first
    sector (0x1000 bytes) is a header area; file directory entries begin at
    sector 1 and live at sector boundaries.
    """
    files: dict[str, bytes] = {}
    offset = _SECTOR  # skip the empty header sector
    while offset + 3 < len(data):
        entry_type = _read_int(data, offset)
        if entry_type == 2:  # file entry
            name = _read_filename(data, offset + 0x04)
            file_size = _read_int(data, offset + 0x8C)
            # Sector index array at +0x94, terminated by 0.
            data_ptr = offset + 0x94
            sector_count = 0
            sector = _read_int(data, data_ptr)
            chunks: list[bytes] = []
            while sector != 0:
                start = sector * _SECTOR
                if start >= len(data):
                    break  # malformed — sector out of range
                chunks.append(data[start:start + _SECTOR])
                sector_count += 1
                sector = _read_int(data, data_ptr + 4 * sector_count)
            file_data = b"".join(chunks).rstrip(b"\x00")
            if file_size > 0 and file_size < len(file_data):
                file_data = file_data[:file_size]
            if name in files:
                files[name] += file_data
            else:
                files[name] = file_data
        offset += _SECTOR
    return files


def extract_score_gpif(path: str | Path) -> bytes | None:
    """Read a ``.gpx`` file and return the ``score.gpif`` XML bytes.

    Returns ``None`` if the file is not a valid GP6 container or contains no
    ``score.gpif`` entry.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if raw[:4] == _BCFS:
        bcfs = raw
    elif raw[:4] == _BCFZ:
        bcfs = decompress_bcfz(raw)
    else:
        return None
    files = parse_bcfs(bcfs)
    return files.get("score.gpif")


__all__ = ["decompress_bcfz", "parse_bcfs", "extract_score_gpif"]
