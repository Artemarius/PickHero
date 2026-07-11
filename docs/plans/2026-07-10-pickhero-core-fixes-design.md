# PickHero Core Correctness Fixes — Design

## Motivation

The assessment identified four P0 correctness defects that make detector measurements unreliable:
1. Verification ring buffer silently corrupts sample/timebase correspondence under lock contention.
2. Segmented clock lacks proper inverse mapping and grows unbounded in wait mode.
3. Three competing scoring paths produce inconsistent judgments per event.
4. Guided practice reads stats after matcher reset, seeing zero accuracy.

Also: calibration offset, chord_fft_size, and latency_mode config values are stored but not wired into gameplay paths.

## Approach

Two milestones to keep each change focused and reviewable:

**M1**: Ring buffer, clock, guided-practice fix, scaffolding wiring — independent of matcher internals.  
**M2**: Unified event state machine replacing the three scoring paths.

---

## M1 — Ring, Clock, Guided Practice, Scaffolding

### 1. SPSC Verification Ring (input.py)

**Current defect**: Non-blocking lock at input.py:152 can silently drop writes while `_detector_sample_offset` continues advancing. `get_window_between()` then reads from a corrupted timebase.

**Design**: Keep the numpy ring for contiguous memory layout, but replace the lock-based write with a short-duration snapshot lock. SoundDevice delivers variable-sized `mono` blocks (host buffer size, e.g. 256/512/1024). Slice each into `hop_size` chunks before writing.

```
_ring: np.ndarray[float32]              # pre-allocated, ring_len samples
_ring_len: int                          # constant after init
_ring_hop: int                          # fixed chunk size (= hop_size)
_ring_write_pos: int                    # absolute sample position (producer write)
_ring_snapshot_lock: threading.Lock     # held only during snapshot copy
_ring_overrun: bool                     # producer lapped consumer
```

Callback (producer — PortAudio native thread):
- Slice `mono` into `hop_size` chunks. For each chunk:
  - Wrap into ring at `samples = min(hop_size, remaining); pos = _ring_write_pos % _ring_len; copy into _ring[pos:pos+samples]` (handle wrap via two slices).
  - `_ring_write_pos += samples` — plain store.
- No lock in the callback. The GIL protects `_ring_write_pos` (CPython int assignment is a single pointer store) and numpy writes are C-level stores. The consumer may see a partially updated ring if it reads during a two-slice wrap — the snapshot lock fixes this.

`get_window_between()` (consumer — main thread):
- Acquire `_ring_snapshot_lock` briefly.
  - Read `snapshot_pos = self._ring_write_pos` (GIL-serialized, safe).
  - Copy ring: `ring_copy = self._ring.copy()` (numpy memcpy; callback writes during copy may or may not be visible — results are bounded-stale, which is acceptable for a verifier window).
- Release lock.
- Compute `start_sample, end_sample` in absolute indices.
- If `end_sample > snapshot_pos`: window not fully available → return None.
- If `start_sample < snapshot_pos - _ring_len`: window evicted → return None.
- Extract slice from `ring_copy` at modular indices.

**Overrun**: After each callback write, if `_ring_write_pos - _ring_len > _last_read_pos` (best-effort consumer hint), set `_ring_overrun = True`. Consumer logs it.

**Removed**: The try/acquire(blocking=False) pattern from the callback. `_ring_write_idx` (replaced by absolute `_ring_write_pos`). The silent drop path where lock contention silently loses samples.

### 2. Bounded Segmented Clock (clock.py, scrolling.py)

**Current defect**: `song_to_stream_ms` always uses latest segment; `stream_to_song_ms` iterates backward — not inverses. Wait mode adds a segment every frozen frame.

**Design**:

Each `ClockSegment` stores both song and stream origins. No range end needed — segments are implicitly ordered; the newest is the active one. But we add explicit handling for closed segments:

```
MAX_SEGMENTS = 32
```

`set_segment()`:
- Appends a new segment with the given origins.
- If `len(_segments) > MAX_SEGMENTS`: find the oldest **complete** segment (one that is not the active segment). Drop it and log a warning. Never drop the active segment.

Wait mode (scrolling.py:366-388):
- **On entering freeze**: one segment added, `tempo_factor=0.0`. Stream advances, song frozen — stream → song maps all stream time to the same frozen song position.
- **On leaving freeze**: add one segment with `tempo_factor=self._tempo_factor`. Real song time resumes.
- **No segments added during a frozen frame**. The single freeze segment covers the entire frozen duration.

`song_to_stream_ms(song_ms)`:
- Finds the segment whose song range contains `song_ms` (or the latest segment as fallback).
- `seg.stream_origin_ms + (song_ms - seg.song_origin_ms) / seg.tempo_factor`

`stream_to_song_ms(stream_ms)`:
- Finds the segment whose stream range contains `stream_ms` (iterate reversed, first with `stream_origin_ms <= stream_ms`).
- `seg.song_origin_ms + (stream_ms - seg.stream_origin_ms) * seg.tempo_factor`

These are true inverses for any point within a segment's span.

### 3. Guided Practice Reorder (scrolling.py:514-531)

**Defect**: `self._matcher.reset()` called before `get_statistics()`.

**Fix**: Swap the order — snapshot stats, then reset.

```python
if self._matcher:
    stats = self._matcher.get_statistics()   # snapshot first
    self._matcher.reset()                     # then clear
```

### 4. Scaffolding Wiring (config.py, scrolling.py, verifier_composite.py)

| Field | File | Action |
|---|---|---|
| `audio_offset_ms` | scrolling.py:1642 | Replace `audio_offset_ms=0.0` with `audio_offset_ms=self._config.audio_latency_offset_ms` |
| `chord_fft_size` | verifier_composite.py:34 | Accept `fft_size` parameter, forward to `CQTVerifier(sample_rate, fft_size)` |
| `latency_mode` | input.py:448 | `latency` param reads from `ac.latency_mode` → maps to `sd.latency` compatible value |
| `PitchEngine.set_tab_prior` | input.py | `set_tab_context(midi_set)` calls `self._engine.set_tab_prior(midi_set)` when engine present |
| Inactive preset fields | config.py:82-85 | Add doc-comment marking `multi_label_techniques`, `after_take_analyzer`, `tone_profile_required` as inactive |

---

## M2 — Unified Event State Machine (matcher.py)

Replaces the three scoring paths (`verify_hit_zone`, `process_detected_notes`, `verify_chord_at`) with one state machine.

### Event States

```python
class EventState(Enum):
    PENDING = "pending"              # waiting for evidence
    ATTACKING = "attacking"          # onset detected, awaiting pitch confirmation
    PITCHED = "pitched"             # pitch confirmed, within sustain window
    SUSTAINING = "sustaining"        # active sustain, periodic pitch re-check for long notes
    RELEASED = "released"            # note ended, computing final verdict
    HIT = "hit"                     # final: success
    PARTIAL = "partial"             # final: partial success (chord)
    MISS = "miss"                   # final: failure
```

### Transitions

Technique evidence does not drive state transitions. Only pitch, onset, and timing evidence do.

```
PENDING
  ├─ has_onset AND pitch_matches → PITCHED
  │   (onset + pitch in same window — common case)
  ├─ has_onset AND no_pitch_yet → ATTACKING
  │   (onset seen, pitch needs more time to lock)
  └─ timing_window_expired AND no_onset → MISS

ATTACKING
  ├─ pitch_matches → PITCHED
  ├─ window_expired AND no_pitch → MISS
  └─ sustained_noise_but_no_pitch → MISS

PITCHED
  ├─ note_duration_expired → RELEASED
  ├─ pitch_still_matches (re-checked during sustain) → stays PITCHED
  └─ strong_pitch_contradiction (e.g., octave error confirmed across multiple frames) → MISS

SUSTAINING (optional — for long notes > 500ms)
  └─ note_duration_expired → RELEASED

RELEASED → HIT
  (for chord groups: HIT if >= critical roles present; PARTIAL if root + one other;
   CLOSE if root only; MISS if no root)
```

Technique verdicts are computed in parallel and attached as metadata to the terminal state. They never block or redirect a transition.

### Chord Handling

Chord verdict computed at the PITCHED→RELEASED transition from unique pitch classes:

```python
@dataclass
class ChordVerdict:
    root_ok: bool
    third_ok: bool | None       # None = no third in chord
    seventh_ok: bool | None
    fifth_ok: bool               # at least one fifth present
    extra_pitch_classes: int      # pitch classes detected but not in chord
```

Verdict: HIT if all critical roles present + no extra classes. PARTIAL if critical roles present but duplicate notes missing. CLOSE if root + one other role. MISS otherwise.

### Evidence Sources

The state machine accepts observations from:
- **YIN/detected notes**: pitch candidates with confidence
- **Onset detector**: attack timing
- **Spectral verifier**: harmonic support, anti-aliasing
- **CQT verifier**: chord energy, strum onset
- **Technique verifiers**: bend/slide/vibrato/etc. (separate verdicts, not pitch veto)

### Technique Policy

Technique evidence produces a separate `TechniqueVerdict` per technique (present, uncertain, absent). **Technique never affects the base note verdict.** Correct pitch/onset → base note HIT.

- `uncertain` → no effect (forgiveness for hard-to-detect techniques like vibrato on short windows)
- `present` → technique quality bonus in per-note accuracy score
- `absent` → technique quality penalty in per-note accuracy score

Technique verdicts feed per-technique skill tracking and practice recommendations, not the frame score. The only exception: when technique evidence directly contradicts pitch (e.g., expected harmonic produces wrong fundamental), the pitch verdict handles that — technique plays no role.

### Integration

Replaces `scrolling.py:439-467` with:

```python
results = self._matcher.advance_state_machine(
    playback_ms=self._playback_ms,
    audio_window=audio_window,
    detected_notes=detected,
    chord_result=chord_result,
)
```

The `advance_state_machine` method processes all evidence and returns new terminal-state results.

### Tie Note Support

`loader.py` stops skipping tie notes. Tie notes enter the state machine at `PITCHED` (no onset required, sustain from parent note). The parent and tie are linked in the state machine — both reach HIT when the parent is correctly sustained through the tie duration.

---

## Verification

| Area | Test Strategy |
|---|---|
| Ring | Synthetic: fast producer push, slow consumer read. Verify data integrity, detect overruns, confirm no corruption. |
| Clock | Round-trip: `song_to_stream_ms(stream_to_song_ms(t)) == t` after loop/seek/freeze. Verify bounded growth. |
| Guided practice | Matcher with known stats → call loop boundary → verify pre-reset values returned. |
| State machine | Unit: each transition path with synthetic evidence. Chord verdict with every combination. |
| Scaffolding | Assert calibration wired, `ChordVerifier` receives non-default fft_size. |

M1 landing criteria: all existing tests pass, synthetic ring test passes, clock round-trip tests pass, guided-practice test passes.  
M2 landing criteria: state machine produces equivalent-or-better results on corpus, technique no longer vetoes notes.
