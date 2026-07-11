"""Tests for pickhero.tabs.gpx — GP6 .gpx container decompression."""

import struct

import pytest

from pickhero.tabs.gpx import (
    _BitReader,
    decompress_bcfz,
    parse_bcfs,
    extract_score_gpif,
    _read_int,
    _read_filename,
    _SECTOR,
    _BCFZ,
    _BCFS,
)


class TestBitReader:
    """MSB-first bit reader over a bytes buffer."""

    def test_read_bit_zero(self):
        reader = _BitReader(b"\x00")
        assert reader.read_bit() == 0

    def test_read_bit_one(self):
        reader = _BitReader(b"\xff")
        assert reader.read_bit() == 1

    def test_read_bit_msb_first(self):
        # 0b10000000 → first bit is 1, rest are 0
        reader = _BitReader(b"\x80")
        assert reader.read_bit() == 1
        assert reader.read_bit() == 0
        assert reader.read_bit() == 0

    def test_read_bits_count(self):
        # 0b10101010 = 170
        reader = _BitReader(b"\xaa")
        assert reader.read_bits(4) == 0b1010

    def test_read_bits_eight_bits(self):
        reader = _BitReader(b"\xff")
        assert reader.read_bits(8) == 0xFF

    def test_read_bits_zero_count(self):
        reader = _BitReader(b"\xff")
        assert reader.read_bits(0) == 0

    def test_read_bits_reversed(self):
        # 0b11000000 → read 3 bits reversed
        # Normal: bits are 1,1,0 → 0b110 = 6
        # Reversed: bits are 0,1,1 → 0b011 = 3
        reader = _BitReader(b"\xc0")
        result = reader.read_bits(3, reversed_=True)
        # MSB-first reads 1,1,0. reversed_ = [0,1,1]. value = 0*4+1*2+1 = 3
        assert result == 3

    def test_read_bytes(self):
        reader = _BitReader(b"\x01\x02")
        assert reader.read_bytes(2) == b"\x01\x02"

    def test_read_bytes_alignment(self):
        # After reading 1 bit, 7 bits remain in byte 0, then byte 1
        reader = _BitReader(b"\x80\x01")
        reader.read_bit()  # consume the 1 bit
        # Remaining bits: 0000000 0000001 → read_bytes(2) reads 8+8 bits
        result = reader.read_bytes(2)
        assert len(result) == 2

    def test_past_end_returns_zeros(self):
        reader = _BitReader(b"")
        assert reader.read_bit() == 0
        assert reader.read_bits(8) == 0


class TestReadInt:
    def test_read_int_positive(self):
        data = struct.pack("<i", 42)
        assert _read_int(data, 0) == 42

    def test_read_int_with_offset(self):
        data = b"\x00\x00" + struct.pack("<i", 99)
        assert _read_int(data, 2) == 99

    def test_read_int_negative(self):
        data = struct.pack("<i", -1)
        assert _read_int(data, 0) == -1


class TestReadFilename:
    def test_simple_filename(self):
        name = b"score.gpif\x00" + b"\x00" * 50
        assert _read_filename(name, 0) == "score.gpif"

    def test_stops_at_null(self):
        name = b"file.xml\x00garbage"
        assert _read_filename(name, 0) == "file.xml"

    def test_with_offset(self):
        data = b"\x00\x00MyFile\x00\x00"
        assert _read_filename(data, 2) == "MyFile"

    def test_with_length(self):
        data = b"ABCDEFGHIJ"
        assert _read_filename(data, 0, length=5) == "ABCDE"

    def test_empty_string(self):
        data = b"\x00\x00"
        assert _read_filename(data, 0) == ""


class TestDecompressBcfz:
    def test_already_bcfs_passthrough(self):
        """If data starts with BCFS magic, return as-is."""
        data = _BCFS + b"\x00" * 100
        result = decompress_bcfz(data)
        assert result == data

    def test_empty_after_stripping_magic(self):
        """BCFZ magic stripped but no size header → struct.error."""
        with pytest.raises(struct.error):
            decompress_bcfz(b"")

    def test_bcfz_magic_consumed(self):
        """BCFZ magic is detected and stripped automatically."""
        # Build a minimal BCFZ: magic + expected_size=0 → returns b""
        data = _BCFZ + struct.pack("<i", 0)
        result = decompress_bcfz(data)
        assert result == b""

    def test_zero_expected_size_returns_empty(self):
        """expected_size=0 after magic → returns empty."""
        data = struct.pack("<i", 0) + b"\x00" * 10
        result = decompress_bcfz(data)
        assert result == b""

    def test_literal_only(self):
        """A BCFZ stream with only literal runs (flag=0)."""
        # Build: expected_size=3, then a literal flag (0 bit), size=3 (2 bits reversed)
        # Literal flag = 0, size = 3 (binary 11, but reversed_ → bits 1,1 → value 3)
        # Then 3 literal bytes
        # Byte layout after the 4-byte size:
        # Bit 0: flag=0
        # Bits 1-2: size=3 (reversed_: read 2 bits 1,1 → reversed → 1,1 → 3)
        # Bits 3-10: 3 bytes of data (24 bits)
        # Let's construct: flag=0, size=3, data = b"ABC"
        # Byte 0: 0|11|00001 → 0b01100001 = 0x61? No, let's be more careful.
        # Bit positions:
        # 0: flag = 0
        # 1-2: size reversed_ (2 bits) → need value 3 → bits read are 1,1 → reversed stays 1,1 → value = 3
        # 3-10: byte 0 of literal (8 bits)
        # 11-18: byte 1
        # 19-26: byte 2
        # 0b0_11_01000001 = 0b01101000 01...
        # Let's use _BitReader to construct this
        expected_size = 3
        # Build bitstream manually
        bits = []
        bits.append(0)  # flag = literal
        bits.extend([1, 1])  # size = 3 (reversed: read 1,1 → value 3)
        # Literal bytes: A=0x41=01000001, B=0x42=01000010, C=0x43=01000011
        for byte_val in (0x41, 0x42, 0x43):
            for bit_pos in range(7, -1, -1):
                bits.append((byte_val >> bit_pos) & 1)
        # Pack bits into bytes
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
                else:
                    byte <<= 1
            out.append(byte)
        data = struct.pack("<i", expected_size) + bytes(out)
        result = decompress_bcfz(data)
        assert result == b"ABC"


class TestParseBcfs:
    def test_empty_data(self):
        """Empty BCFS image → no files."""
        result = parse_bcfs(b"")
        assert result == {}

    def test_header_only(self):
        """Just a header sector (0x1000 zeros) → no files."""
        data = b"\x00" * _SECTOR
        result = parse_bcfs(data)
        assert result == {}

    def test_single_file_entry(self):
        """Build a minimal BCFS with one file entry."""
        # Header sector (all zeros) + one file entry sector
        header = b"\x00" * _SECTOR
        # File entry: type=2 at offset 0, name at offset 4, file_size at 0x8C,
        # sector indices at 0x94
        entry = bytearray(_SECTOR)
        struct.pack_into("<i", entry, 0, 2)  # entry_type = 2
        name = b"score.gpif\x00"
        entry[4:4 + len(name)] = name
        file_content = b"<XML>test</XML>"
        struct.pack_into("<i", entry, 0x8C, len(file_content))
        # Point to sector 2 (index 2, since 0=header, 1=entry)
        struct.pack_into("<i", entry, 0x94, 2)
        struct.pack_into("<i", entry, 0x98, 0)  # terminate sector list
        # Data sector
        data_sector = bytearray(_SECTOR)
        data_sector[:len(file_content)] = file_content
        data = header + bytes(entry) + bytes(data_sector)
        result = parse_bcfs(data)
        assert "score.gpif" in result
        assert result["score.gpif"] == file_content


class TestExtractScoreGpif:
    def test_nonexistent_file(self, tmp_path):
        """Non-existent file → None."""
        assert extract_score_gpif(tmp_path / "nonexistent.gpx") is None

    def test_invalid_magic(self, tmp_path):
        """File without BCFZ/BCFS magic → None."""
        bad_file = tmp_path / "bad.gpx"
        bad_file.write_bytes(b"XXXX" + b"\x00" * 100)
        assert extract_score_gpif(bad_file) is None

    def test_bcfs_passthrough(self, tmp_path):
        """A raw BCFS file (already decompressed) is parsed directly."""
        # Build a minimal BCFS with score.gpif
        header = b"\x00" * _SECTOR
        entry = bytearray(_SECTOR)
        struct.pack_into("<i", entry, 0, 2)
        name = b"score.gpif\x00"
        entry[4:4 + len(name)] = name
        xml = b"<GPIF><score></score></GPIF>"
        struct.pack_into("<i", entry, 0x8C, len(xml))
        struct.pack_into("<i", entry, 0x94, 2)
        struct.pack_into("<i", entry, 0x98, 0)
        data_sector = bytearray(_SECTOR)
        data_sector[:len(xml)] = xml
        bcfs = _BCFS + header[4:] + bytes(entry) + bytes(data_sector)
        gpx_file = tmp_path / "test.gpx"
        gpx_file.write_bytes(bcfs)
        result = extract_score_gpif(gpx_file)
        assert result == xml

    def test_pathlib_path(self, tmp_path):
        """Accepts pathlib.Path objects."""
        from pathlib import Path
        bad_file = Path(tmp_path) / "nonexistent.gpx"
        assert extract_score_gpif(bad_file) is None

    def test_no_score_gpif_returns_none(self, tmp_path):
        """BCFS without score.gpif entry → None."""
        header = b"\x00" * _SECTOR
        # Build an entry for a different file
        entry = bytearray(_SECTOR)
        struct.pack_into("<i", entry, 0, 2)
        name = b"other.xml\x00"
        entry[4:4 + len(name)] = name
        content = b"<other/>"
        struct.pack_into("<i", entry, 0x8C, len(content))
        struct.pack_into("<i", entry, 0x94, 2)
        struct.pack_into("<i", entry, 0x98, 0)
        data_sector = bytearray(_SECTOR)
        data_sector[:len(content)] = content
        bcfs = _BCFS + header[4:] + bytes(entry) + bytes(data_sector)
        gpx_file = tmp_path / "no_score.gpx"
        gpx_file.write_bytes(bcfs)
        result = extract_score_gpif(gpx_file)
        assert result is None


class TestConstants:
    def test_sector_size(self):
        assert _SECTOR == 0x1000

    def test_magic_values(self):
        assert _BCFZ == b"BCFZ"
        assert _BCFS == b"BCFS"
