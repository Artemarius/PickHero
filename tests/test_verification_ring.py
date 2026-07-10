"""Tests for the numpy SPSC verification ring design."""
import numpy as np
import threading


def test_numpy_ring_data_integrity():
    """Fixed-hop chunks wrapped into ring — data reads back in order."""
    ring_len = 4096
    hop = 256
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0

    # Write 32 chunks (each hop-sized, total = 32*256 = 8192 samples, wraps ~2x)
    for i in range(32):
        chunk = np.arange(i * hop, (i + 1) * hop, dtype=np.float32)
        pos = write_pos % ring_len
        end = pos + hop
        if end <= ring_len:
            ring[pos:end] = chunk
        else:
            first = ring_len - pos
            ring[pos:] = chunk[:first]
            ring[:end - ring_len] = chunk[first:]
        write_pos += hop

    # Read back last ring_len samples starting from write_pos - ring_len
    read_start = write_pos - ring_len
    result = np.zeros(ring_len, dtype=np.float32)
    for i in range(ring_len):
        result[i] = ring[(read_start + i) % ring_len]
    expected = np.arange(read_start, write_pos, dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_numpy_ring_snapshot_consistency():
    """Snapshot under lock returns a self-consistent copy."""
    ring_len = 2048
    hop = 256
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0
    lock = threading.Lock()

    for i in range(4):
        chunk = np.full(hop, float(i + 1), dtype=np.float32)
        pos = write_pos % ring_len
        end = pos + hop
        if end <= ring_len:
            ring[pos:end] = chunk
        else:
            first = ring_len - pos
            ring[pos:] = chunk[:first]
            ring[:end - ring_len] = chunk[first:]
        write_pos += hop

    with lock:
        snap_pos = write_pos
        ring_copy = ring.copy()

    for i in range(ring_len):
        src_idx = (snap_pos - ring_len + i) % ring_len
        val = ring_copy[src_idx]
        assert val in (0.0, 1.0, 2.0, 3.0, 4.0), f"Unexpected value {val} at idx {i}"


def test_numpy_ring_window_extraction():
    """Extract [start, end) sample range from ring with wrap."""
    ring_len = 1024
    hop = 256
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0

    for i in range(8):
        chunk = np.arange(i * hop, (i + 1) * hop, dtype=np.float32)
        pos = write_pos % ring_len
        end = pos + hop
        if end <= ring_len:
            ring[pos:end] = chunk
        else:
            first = ring_len - pos
            ring[pos:] = chunk[:first]
            ring[:end - ring_len] = chunk[first:]
        write_pos += hop

    snap_pos = write_pos  # 2048
    ring_start = snap_pos - ring_len  # 1024

    # Window [1500:1800) should be fully in the ring
    start_ok = 1500
    end_ok = 1800
    assert start_ok >= ring_start and end_ok <= snap_pos

    window = np.zeros(end_ok - start_ok, dtype=np.float32)
    for i in range(end_ok - start_ok):
        window[i] = ring[(start_ok + i) % ring_len]

    expected = np.arange(1500, 1800, dtype=np.float32)
    np.testing.assert_array_equal(window, expected)
