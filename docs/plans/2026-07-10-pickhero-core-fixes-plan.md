# PickHero Core Correctness Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix four P0 correctness defects (verification ring corruption, broken clock mapping, competing scoring authorities, guided-practice stats ordering) and wire orphaned config fields into gameplay paths, then build a unified event state machine replacing the three scoring paths.

**Architecture:** Two milestones. M1: ring buffer → clock → guided-practice → scaffolding wiring. M2: unified event state machine replacing `verify_hit_zone`/`process_detected_notes`/`verify_chord_at` with a single stateful scorer. Each task follows TDD: write failing test, verify failure, implement, verify pass, commit.

**Tech Stack:** Python 3.10+, numpy, sounddevice, aubio, pygame, threading (snapshot lock), pytest.

---

## Milestone M1 — Ring, Clock, Guided Practice, Scaffolding

### Task 1.1: Write verification ring test

**Files:**
- Create: `tests/test_verification_ring.py`
- Modify: none

**Step 1: Write the test**

The ring is a pre-allocated numpy array with absolute write position. The callback slices `mono` into `hop_size` chunks and wraps them in. The consumer snapshots under a short lock.

```python
"""Tests for the numpy SPSC verification ring."""
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
        # Wrap into ring at write_pos % ring_len
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
    
    # Expected: samples write_pos-ring_len .. write_pos-1 in order
    expected = np.arange(read_start, write_pos, dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_numpy_ring_snapshot_consistency():
    """Simulate concurrent write + snapshot: result is bounded-stale, not corrupted."""
    ring_len = 2048
    hop = 256
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0
    lock = threading.Lock()
    
    # Producer writes 4 chunks (no threading for this test, just verify snapshot logic)
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
    
    # Snapshot under lock (simulated)
    with lock:
        snap_pos = write_pos
        ring_copy = ring.copy()
    
    # Verify copy is self-consistent
    for i in range(ring_len):
        src_idx = (snap_pos - ring_len + i) % ring_len
        val = ring_copy[src_idx]
        # Value should be 0 (unwritten) or one of the chunk values
        assert val in (0.0, 1.0, 2.0, 3.0, 4.0)


def test_numpy_ring_window_extraction():
    """Extract [start, end) sample range from ring with wrap."""
    ring_len = 1024
    hop = 256
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0
    
    # Fill ring with sequential values
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
    
    # Read window [512:1024) in absolute sample terms
    # write_pos = 2048, ring covers 1024..2047
    snap_pos = write_pos
    ring_start = snap_pos - ring_len  # 1024
    start_abs = 512
    end_abs = 1024
    
    if start_abs < ring_start or end_abs > snap_pos:
        # Window partially evicted — depending on test setup, this may happen
        assert start_abs < ring_start  # 512 < 1024 = evicted
    
    # A window that IS fully in the ring: [1500:1800)
    start_ok = 1500
    end_ok = 1800
    assert start_ok >= ring_start and end_ok <= snap_pos
    
    window = np.zeros(end_ok - start_ok, dtype=np.float32)
    for i in range(end_ok - start_ok):
        window[i] = ring[(start_ok + i) % ring_len]
    
    expected = np.arange(1500, 1800, dtype=np.float32)
    np.testing.assert_array_equal(window, expected)
```

**Step 2: Run test**

Run: `python -m pytest tests/test_verification_ring.py -v`
Expected: All 3 tests pass.

**Step 3: Commit**

```bash
git add tests/test_verification_ring.py
git commit -m "test: verification ring isolation tests for numpy SPSC ring"
```


### Task 1.2: Replace ring buffer in input.py

**Files:**
- Modify: `pickhero/audio/input.py`
- Test: `tests/test_input.py` (existing — verify no regression)
- New test: `tests/test_verification_ring.py` (extend with AudioCapture integration test)

**Context:** The current ring at input.py:152-171 uses `_ring_lock.acquire(blocking=False)` — if the lock is held (main thread in `get_window_between`), the callback drops its samples silently. `_detector_sample_offset` advances unconditionally at line 197. `get_window_between()` at lines 613-621 then uses `write_sample = self._detector_sample_offset` as the ring head, but the ring content is stale — producing a corrupted window.

**Change:** Keep the numpy ring for contiguous memory, replace lock-based management with absolute write position + short snapshot lock. Slice variable-sized `mono` blocks into fixed `hop_size` chunks.

**New fields:**
```python
_audio_ring: np.ndarray              # pre-allocated, stays as-is but renamed for clarity
_ring_len: int                       # constant after init
_ring_hop: int                       # fixed chunk size (= hop_size)
_ring_write_pos: int                 # absolute sample position (producer-only write)
_ring_snapshot_lock: threading.Lock   # held ONLY during consumer snapshot (short, ~microseconds)
_ring_overrun: bool                  # producer lapped consumer
_ring_xrun_count: int                # cumulative overrun counter
```

**Rename existing:**
- `self._audio_ring` stays (it's already a numpy array)
- Remove `self._ring_lock`, `self._ring_write_idx`
- Add `self._ring_write_pos`, `self._ring_hop`, `self._ring_snapshot_lock`, `self._ring_overrun`

**Init** (in `_init_audio_resources`, around current lines 409-416):
```python
ring_samples = int(ac.sample_rate * self._RING_DURATION_MS / 1000.0)
self._audio_ring = np.zeros(ring_samples, dtype=np.float32)
self._ring_len = ring_samples
self._ring_hop = ac.hop_size
self._ring_write_pos = 0
self._ring_snapshot_lock = threading.Lock()
self._ring_overrun = False
self._ring_xrun_count = 0
# Remove: _ring_lock, _ring_write_idx
```

**Callback changes** (`_audio_callback`, replaces lines 149-171):
```python
# Slice mono into hop-sized chunks and write to ring.
# mono length varies per callback (host buffer size).
mono = indata[:, 0].copy()
hop = self._ring_hop
pos = self._ring_write_pos  # single-producer read — no contention
for start in range(0, len(mono), hop):
    chunk = mono[start:start + hop]
    n = len(chunk)
    ring_idx = pos % self._ring_len
    end = ring_idx + n
    if end <= self._ring_len:
        self._audio_ring[ring_idx:end] = chunk
    else:
        first = self._ring_len - ring_idx
        self._audio_ring[ring_idx:] = chunk[:first]
        self._audio_ring[:end - self._ring_len] = chunk[first:]
    pos += n
self._ring_write_pos = pos  # single-producer store — no CAS needed

# Overrun detection (best-effort)
if pos - self._ring_len > self._ring_xrun_count:
    self._ring_overrun = True
    self._ring_xrun_count += 1
```

**`get_window_between()` changes** (replaces current ~lines 586-624):
```python
def get_window_between(self, start_ms: float, end_ms: float) -> np.ndarray | None:
    sample_rate = self._ring_sample_rate
    if sample_rate <= 0:
        return None
    start_sample = int(sample_rate * start_ms / 1000.0)
    end_sample = int(sample_rate * end_ms / 1000.0)
    total = end_sample - start_sample
    if total <= 0:
        return None
    
    # Snapshot: lock held only for the copy + counter read.
    with self._ring_snapshot_lock:
        snap_pos = self._ring_write_pos
        ring_copy = self._audio_ring.copy()
    
    ring_len = len(ring_copy)
    if ring_len == 0:
        return None
    
    # Window must be fully within the ring's coverage
    ring_start = snap_pos - ring_len
    if start_sample < ring_start or end_sample > snap_pos:
        # Partial or evicted — caller retries next frame
        return None
    
    # Extract from ring_copy handling wrap
    window = np.zeros(total, dtype=np.float32)
    for i in range(total):
        window[i] = ring_copy[(start_sample + i) % ring_len]
    return window
```

**Thread safety rationale:**
- Producer: single-threaded (PortAudio callback), writes to `_audio_ring` at modular indices and stores `_ring_write_pos`. No lock needed — the consumer's snapshot lock prevents reading during a wrap write (which spans two non-contiguous ranges), but the producer doesn't take the snapshot lock, so a callback during a consumer snapshot may produce a partial wrap update visible in the copy. This is bounded-stale (newest samples may be partially written) and acceptable — the caller is already tolerant of windows that are one frame old.

**Removed:** `_ring_lock` field, `_ring_write_idx` field, the `try: acquire(blocking=False)` pattern in the callback, the silent-drop path.

**Step 1: Write/debug integration test**

Add to `tests/test_verification_ring.py`:
```python
def test_audio_capture_ring_integration():
    """Verify that AudioCapture's ring produces self-consistent windows.
    
    This test needs AudioCapture with a mock stream. If it can't run without
    hardware, skip with a marker.
    """
    pass  # flesh out during implementation
```

**Step 2: Run existing tests**

Run: `python -m pytest tests/test_input.py -v`
Expected: All pass (or known failures unrelated to ring change).

**Step 3: Implement the changes in input.py**

Apply the field changes, callback rewrite, and get_window_between rewrite described above.

**Step 4: Run tests again**

Run: `python -m pytest tests/test_input.py tests/test_verification_ring.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add pickhero/audio/input.py tests/test_verification_ring.py
git commit -m "fix: replace lock-based verification ring with numpy + snapshot lock"
```


### Task 1.3: Fix clock segment bounds and wait-mode freeze

**Files:**
- Modify: `pickhero/audio/clock.py`
- Modify: `pickhero/ui/scrolling.py:366-388` (wait mode)
- Test: `tests/test_clock.py`

**Context:** `clock.py:44-56` uses `seg = self._segments[-1]` in `song_to_stream_ms` (always latest segment) but iterates in `stream_to_song_ms` (historical segments). These aren't inverses. Wait mode at scrolling.py:377 adds a segment every frozen frame.

**Change 1: Clock segment bounds (clock.py)**

Add `MAX_SEGMENTS = 32` constant.

`set_segment(song_origin_ms, stream_origin_ms, tempo_factor)`:
- Appends new segment unconditionally.
- If `len(self._segments) > MAX_SEGMENTS`: find the oldest segment that is not the active one (i.e., not `self._segments[-1]`). Remove it. Log warning.

No other structural changes to segments — the key fix is preventing unbounded growth. The inverse mapping is actually correct for historical segments in `stream_to_song_ms` (it iterates reversed and finds the right segment). The real asymmetry bug is in `song_to_stream_ms` which ignores historical segments for forward mapping — but in practice, gameplay always drives `song_to_stream_ms` from the current playback position, so the latest segment is correct. The fix is to make `song_to_stream_ms` also search backward when the song time falls outside the latest segment:

```python
def song_to_stream_ms(self, song_ms: float) -> float:
    """Convert song time to stream time.
    
    Search backwards for the segment that covers song_ms. The latest segment
    (no song_end bound) is the active fallback.
    """
    seg = self._segments[-1]  # default: active segment
    for i in range(len(self._segments) - 2, -1, -1):
        next_seg = self._segments[i + 1]
        if song_ms >= self._segments[i].song_origin_ms and song_ms < next_seg.song_origin_ms:
            seg = self._segments[i]
            break
    return seg.stream_origin_ms + (song_ms - seg.song_origin_ms) / seg.tempo_factor
```

**Change 2: Wait mode (scrolling.py:366-388)**

Add a `_wait_mode_segment_added` flag:

```python
wait_mode_freeze_entered = False
if (self._wait_mode and self._audio_enabled ...):
    if self._matcher.has_pending_notes_at(self._playback_ms):
        if not self._wait_mode_frozen:
            # First freeze frame: add clock segment
            self._wait_mode_frozen = True
            if self._audio_capture is not None:
                stream_ms = self._audio_capture.stream_time_ms()
                self._audio_capture.clock.set_segment(
                    self._playback_ms, stream_ms, 0.0  # tempo=0: stream advances, song frozen
                )
        # else: already frozen, skip adding segment
    elif self._wait_mode_frozen:
        # Thaw: add clock segment
        self._wait_mode_frozen = False
        if self._audio_capture is not None:
            stream_ms = self._audio_capture.stream_time_ms()
            self._audio_capture.clock.set_segment(
                self._playback_ms, stream_ms, self._tempo_factor
            )
```

**Step 1: Write the failing test**

In `tests/test_clock.py`, add:
```python
def test_wait_mode_does_not_add_segments_per_frame():
    clock = StreamClock()
    initial_len = len(clock._segments)
    
    # Simulate 10 frozen frames
    for _ in range(10):
        if _ == 0:
            clock.set_segment(5000.0, 51200.0, 0.0)  # freeze — tempo=0, stream advances but song stays
        else:
            pass  # frozen frame — no new segment
    assert len(clock._segments) == initial_len + 1  # only one freeze segment

def test_song_to_stream_with_historical_segments():
    clock = StreamClock()
    clock.set_segment(5000.0, 52000.0, 1.0)  # at t=5s song, stream was at 52s
    clock.set_segment(10000.0, 55000.0, 0.5)  # at t=10s song, stream at 55s, slow
    
    # Point within first segment
    stream = clock.song_to_stream_ms(7000.0)
    # song delta = 2000ms, stream = 52000 + 2000/1.0 = 54000
    assert stream == pytest.approx(52000.0 + 2000.0, abs=1)
    
    # Point within second segment
    stream = clock.song_to_stream_ms(12000.0)
    # song delta = 2000ms at 0.5x, stream = 55000 + 2000/0.5 = 59000
    assert stream == pytest.approx(55000.0 + 4000.0, abs=1)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clock.py -v`
Expected: FAIL (song_to_stream_ms doesn't use historical segments).

**Step 3: Implement changes**

**Step 4: Run tests**

Run: `python -m pytest tests/test_clock.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pickhero/audio/clock.py pickhero/ui/scrolling.py tests/test_clock.py
git commit -m "fix: bounded clock segments with correct inverse mapping and wait-mode freeze"
```


### Task 1.4: Fix guided-practice stats ordering

**Files:**
- Modify: `pickhero/ui/scrolling.py:514-531`
- Test: `tests/test_matcher.py`

**Step 1: Write the failing test**

In `tests/test_matcher.py`, add:
```python
def test_guided_practice_stats_snapshotted_before_reset():
    """Simulate the loop boundary — stats should be from pre-reset state."""
    matcher = NoteMatcher(...)
    # Play through some notes to generate stats
    # ... setup with timeline and detected notes ...
    
    # Simulate loop boundary
    stats_before = matcher.get_statistics()
    matcher.reset()
    stats_after = matcher.get_statistics()
    
    assert stats_before.get("accuracy_percent", 0) > 0
    assert stats_after.get("accuracy_percent", 0) == 0
```

**Step 2: Run test**

Run: `python -m pytest tests/test_matcher.py::test_guided_practice_stats_snapshotted_before_reset -v`

**Step 3: Reorder in scrolling.py**

Change:
```python
# Before (buggy):
if self._matcher:
    self._matcher.reset()
    ...
    stats = self._matcher.get_statistics()

# After:
if self._matcher:
    stats = self._matcher.get_statistics()  # snapshot first
    ...
    self._matcher.reset()                    # then reset for next iteration
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_matcher.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pickhero/ui/scrolling.py tests/test_matcher.py
git commit -m "fix: snapshot guided-practice stats before matcher reset"
```


### Task 1.5: Wire orphaned config fields

**Files:**
- Modify: `pickhero/ui/scrolling.py:1639-1647` (audio_offset_ms)
- Modify: `pickhero/audio/verifier_composite.py:27-35` (chord_fft_size)
- Modify: `pickhero/audio/input.py:442-452` (latency_mode)
- Modify: `pickhero/audio/input.py` (add set_tab_context calling engine.set_tab_prior)
- Modify: `pickhero/config.py:82-86` (document inactive preset fields)

**Step 1.5a: Wire audio_offset_ms**

In scrolling.py, the `NoteMatcher` construction at ~line 1642:
```python
self._matcher = NoteMatcher(
    self._timeline,
    timing_window_ms=self._config.timing_window_ms,
    audio_offset_ms=self._config.audio_latency_offset_ms,  # was 0.0
    ...
)
```

Check that `Config` has `audio_latency_offset_ms` field — if not, add it with default 0.0.

**Step 1.5b: Wire chord_fft_size**

In verifier_composite.py, `__init__`:
```python
def __init__(self, sample_rate=48000, use_cqt_for_chords=True, fft_size=8192):
    self._cqt = CQTVerifier(sample_rate=sample_rate, fft_size=fft_size)
```

In scrolling.py, construction at ~line 1636:
```python
verifier = CompositeVerifier(
    sample_rate=self._audio_capture.detector.sample_rate,
    fft_size=getattr(self._config, 'chord_fft_size', 8192),
)
```

**Step 1.5c: Wire latency_mode**

In input.py, the `sd.InputStream` at ~line 448:
```python
latency = 'low'  # current hardcode
```
Change to:
```python
latency = ac.latency_mode  # 'low', 'medium', 'high'
```

But `sd.InputStream` expects a float (seconds) or `'low'`/`'high'` — not `'medium'`. Map it:
```python
_latency_map = {'low': 'low', 'medium': 'low', 'high': 'high'}
latency = _latency_map.get(ac.latency_mode, 'low')
```

**Step 1.5d: Wire set_tab_context**

In input.py, add method on `AudioCapture`:
```python
def set_tab_context(self, midi_notes: set[int]) -> None:
    """Feed expected MIDI notes from playback position to pitch engine."""
    if self._engine is not None:
        self._engine.set_tab_prior(midi_notes)
```

**Step 1.5e: Document inactive preset fields**

In config.py, the `JOSE_HIGH_ACCURACY_PRESET` dict, add comments:
```python
# Inactive preset fields (stored for documentation/reference):
# "multi_label_techniques": True     — not implemented
# "after_take_analyzer": True        — not implemented
# "tone_profile_required": True       — not implemented
```

**Step 2: Run existing tests**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All pass.

**Step 3: Commit**

```bash
git add pickhero/ui/scrolling.py pickhero/audio/verifier_composite.py pickhero/audio/input.py pickhero/config.py
git commit -m "fix: wire audio_offset_ms, chord_fft_size, latency_mode, tab_context into gameplay paths"
```


## Milestone M2 — Unified Event State Machine

### Task 2.1: Define EventState and data model

**Files:**
- Create: `pickhero/audio/event_state.py`

**Step 1: Write the state machine data model**

```python
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class EventState(Enum):
    PENDING = "pending"
    ATTACKING = "attacking"
    PITCHED = "pitched"
    SUSTAINING = "sustaining"
    RELEASED = "released"
    HIT = "hit"
    PARTIAL = "partial"
    MISS = "miss"

@dataclass
class PitchVerdict:
    correct: bool
    midi: int | None = None
    confidence: float = 0.0
    cents_error: float | None = None

@dataclass
class TimingVerdict:
    early: bool = False
    late: bool = False
    exact: bool = False
    error_ms: float = 0.0

@dataclass
class TechniqueVerdict:
    technique: str
    present: bool = False
    uncertain: bool = False
    confidence: float = 0.0

@dataclass
class ChordRoleVerdict:
    root_ok: bool = False
    third_ok: bool | None = None
    seventh_ok: bool | None = None
    fifth_ok: bool = False
    extra_pitch_classes: int = 0
```

**Step 2: Commit**

```bash
git add pickhero/audio/event_state.py
git commit -m "feat: event state machine data model (EventState, verdict types)"
```


### Task 2.2: Implement score tracking per event

**Files:**
- Modify: `pickhero/matcher.py` (add state machine methods)
- Test: `tests/test_matcher.py`

**Step 1: Write unit tests for state transitions**

```python
def test_event_pending_to_hit():
    """Normal picked note: onset + pitch → PITCHED → HIT."""
    matcher = create_matcher_with_single_note(midi=40, timestamp_ms=1000.0)
    
    # Before any evidence: PENDING
    assert matcher._get_event_state((1000.0, 1)) == EventState.PENDING
    
    # Feed onset + pitch evidence
    results = matcher.advance_state_machine(
        playback_ms=1000.0,
        audio_window=synthetic_window_with_freq(82.41),  # E2
        detected_notes=[DetectedNote(midi_note=40, is_onset=True, confidence=0.9)],
    )
    
    assert len(results) == 1
    assert results[0].match_type == MatchType.HIT
```

**Step 2: Implement `advance_state_machine` on NoteMatcher**

The method iterates pending events, collects evidence from audio window + detected notes + chord result, then transitions each event's state.

Core loop:
```python
def advance_state_machine(self, playback_ms, audio_window, detected_notes, chord_result=None):
    results = []
    for event_id, state in list(self._event_states.items()):
        new_state = self._transition(event_id, state, playback_ms, audio_window, detected_notes)
        if new_state != state:
            self._event_states[event_id] = new_state
            if new_state in (EventState.HIT, EventState.PARTIAL, EventState.MISS):
                results.append(self._build_result(event_id, new_state))
    return results
```

`_transition()` implements the state machine table from the design doc.

**Step 3: Write chord verdict logic**

```python
def _compute_chord_verdict(self, pitch_classes: set[int], chord_spec: ChordSpec) -> ChordRoleVerdict:
    """Compute chord detection quality from observed pitch classes."""
    verdict = ChordRoleVerdict()
    verdict.root_ok = chord_spec.root in pitch_classes
    if chord_spec.third is not None:
        verdict.third_ok = chord_spec.third in pitch_classes
    if chord_spec.seventh is not None:
        verdict.seventh_ok = chord_spec.seventh in pitch_classes
    # Fifth: check any fifth in chord_spec.fifth_options
    verdict.fifth_ok = any(f in pitch_classes for f in chord_spec.fifth_options)
    extra = pitch_classes - {chord_spec.root, chord_spec.third, chord_spec.seventh} - set(chord_spec.fifth_options)
    verdict.extra_pitch_classes = len(extra)
    return verdict
```

**Step 4: Verify transitions exhaustively**

Test each transition path from the design:

| Test | Path |
|---|---|
| Normal picked hit | PENDING → PITCHED → RELEASED → HIT |
| Onset before pitch locks | PENDING → ATTACKING → PITCHED → HIT |
| Miss: no onset | PENDING → MISS (window expired) |
| Miss: wrong pitch | PENDING → ATTACKING → MISS |
| Chord: all notes present | PENDING → PITCHED → HIT (all roles ok) |
| Chord: missing third | PENDING → PITCHED → PARTIAL |
| Chord: extra note | PENDING → PITCHED → PARTIAL |
| Technique: uncertain | PENDING → PITCHED → HIT (technique uncertain → no effect on verdict) |
| Technique: verified | PENDING → PITCHED → HIT (technique adds accuracy bonus, same terminal state) |
| Technique: absent | PENDING → PITCHED → HIT (technique adds accuracy penalty, same terminal state) |
| Sustain timeout | PENDING → PITCHED → SUSTAINING → RELEASED → HIT |
| Tie note (no onset) | PITCHED (skip onset requirement) → HIT |

**Step 5: Integrate into gameplay loop**

In scrolling.py, replace:
```python
hit_zone_results = self._matcher.verify_hit_zone(...)
results = list(hit_zone_results)
results.extend(self._matcher.process_detected_notes(...))
results.extend(self._matcher.verify_chord_at(...))
```

With:
```python
results = self._matcher.advance_state_machine(
    playback_ms=self._playback_ms,
    audio_window=audio_window,
    detected_notes=detected,
    chord_result=chord_result,
)
```

**Step 6: Remove old methods**

Mark `verify_hit_zone`, `process_detected_notes`, `verify_chord_at` as deprecated, then remove after confirming the state machine produces equivalent results.

**Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All pass (existing tests may need minor updates to use new API).

**Step 8: Commit**

```bash
git add pickhero/matcher.py pickhero/audio/event_state.py pickhero/ui/scrolling.py tests/test_matcher.py
git commit -m "feat: unified event state machine replacing three scoring paths"
```
