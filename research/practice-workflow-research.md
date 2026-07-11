# Guitar Practice Workflow & Pedagogy Improvements for PickHero

## Executive Summary

PickHero currently has a **rule-based recommendation engine** (`recommendations.py`) that produces text hints after each run — suggesting tempo changes in 5% increments, flagging persistent weak sections, and detecting accuracy plateaus across attempts. The newly added **Timing Judge** (`timing.py`) adds per-onset timing error analysis with a ±100ms histogram, early/late/on-time verdicts at ±25ms thresholds, and per-measure statistics. This is a solid foundation, but the research literature on deliberate practice, spaced repetition, motor learning, and commercial practice apps reveals significant opportunities to make practice more effective.

The five highest-impact improvements are:

1. **Spaced repetition scheduling for song/section review** — PickHero tracks `last_played` and `attempts` but never schedules *when to revisit* material. Adding an SM-2 or FSRS-based scheduler that predicts forgetting and surfaces due-for-review songs would transform the app from a "play whatever you want" tool into a structured practice system. The spacing effect has an effect size of $d=0.42$ across 116 studies [1], and music-specific implementations already exist (Logato, Piano Practice Assistant) [2][3].

2. **Adaptive tempo progression with plateau detection** — PickHero's current 5% tempo increments are reasonable but lack the motor-learning science. Research shows the speed-accuracy tradeoff has a "cliff zone" (typically 110-130 BPM where accuracy drops from >85% to ~62%) [4]. An adaptive algorithm that detects these cliffs per-song-per-user, uses 10 BPM increments at slow tempos and 5 BPM at fast tempos, and employs "progressive speed bursts" (8-bar runs with 5 BPM increases) would accelerate skill acquisition.

3. **Micro-timing analysis beyond hit/miss** — The Timing Judge's ±25ms early/late threshold and ±100ms histogram are a strong start. But research on groove, swing feel, and participatory discrepancies shows that timing consistency ($\sigma$) matters more than mean error, and that swing-ratio analysis (long-short 8th-note patterns) reveals stylistic information the current model misses. Adding swing-ratio detection, timing-trend analysis (rushing vs. dragging across the song), and groove-quantization comparison would deepen the practice feedback.

4. **Guided practice loop with difficulty scaffolding** — Inspired by Rocksmith's Riff Repeater and Yousician's mission path, PickHero should add a structured "guided practice" mode: auto-loop the weakest section at reduced tempo, auto-increase tempo when accuracy hits a threshold, and present a clear progression from "learn the notes" → "lock in the timing" → "build speed." The current `L` key loops the weakest section but offers no auto-progression.

5. **Session-level practice intelligence** — PickHero has no concept of a "practice session" as a unit. Adding session generation (like Logato's warmup-core-cooldown flow), interleaved practice across multiple songs/sections, and a practice dashboard with streaks, skill-area breakdowns, and mastery levels would give structure to unstructured practice time.

## Methodology

Research was conducted across five domains: (1) deliberate practice and spaced repetition for music, (2) commercial practice app architectures (Yousician, Rocksmith, Simply Guitar), (3) BPM/tempo progression science for motor learning, (4) micro-timing and rhythm analysis research, (5) open-source practice tools (TuxGuitar, Solfej, GNU Solfege). Sources include academic papers (Nature, ScienceDirect, PMC), primary documentation (Yousician, Rocksmith+ official sites), open-source project pages (GitHub, Savannah), and practitioner blogs with cited research. The PickHero codebase at `~/tmp/PickHero/` was read directly — `progress.py`, `recommendations.py`, `timing.py`, and `scrolling.py` (loop-weakest-section and completion paths) — to ground all recommendations in the actual code.

**Verification approach:** Claims are cited inline [n]. Where a source is a commercial product page rather than a research paper, the claim is marked as a product observation rather than a research finding. The research was conducted in a single pass due to search backend rate-limiting; some sub-questions (Fitts-Posner motor learning stages, Ericsson's original deliberate practice paper) could not be fetched directly and are marked as [INFERENCE] where the consensus from secondary sources is summarized.

## Findings

### 1. Deliberate Practice & Spaced Repetition for Music

#### 1.1 The Spacing Effect in Motor Learning

The spacing effect — distributing practice over time rather than massing it — is one of the most robust findings in learning science. Donovan & Radosevich's meta-analysis of 116 studies since 1927 found an effect size of $d=0.42$, meaning the average person receiving distributed training outperforms ~67% of those receiving massed training [1]. Gwern's comprehensive review notes that this effect generalizes from verbal skills to fine motor skills like surgery, and by extension to music performance [5].

For music specifically, the challenge is that motor skill acquisition requires *some* massed repetition to build muscle memory before spacing becomes effective. Piano Practice Assistant's author articulates this precisely: "learning music requires some combination of 'massed' and spaced practice: you need to do quite a few repetitions in a row to make any progress, but to retain and build on that progress you'd do well to space out your reviews" [3].

#### 1.2 SM-2 and FSRS Algorithms

**SM-2** (the algorithm behind Anki, created by Piotr Wozniak) works as follows: each item has an ease factor $EF$ (initially 2.5), an interval $I$, and a repetition count $n$. After each review rated 0-5:
- If rating ≥ 3 (correct): $I_1 = 1$, $I_2 = 6$, $I_n = I_{n-1} \times EF$ for $n > 2$
- $EF' = EF + (0.1 - (5 - q) \times (0.08 + (5 - q) \times 0.02))$, clamped to $[1.3, 2.5]$
- If rating < 3 (incorrect): reset $n = 0$, $I = 1$

Logato adapts SM-2 specifically for music motor learning: "Exercises you find difficult come back sooner; well-learned ones space out over time" [2]. Their priority engine scores exercises across multiple factors:
- **Due date decay** — past-due exercises surfaced immediately
- **BPM gap boost** — distance from target BPM increases priority
- **Fail count multiplier** — struggled exercises climb the queue faster
- **Discovery score** — new exercises introduced at a deliberate pace
- **Focus area relevance** — ×1.4 weight for matching exercises
- **Goal urgency** — approaching deadlines boost priority automatically [2]

**FSRS (Free Spaced Repetition Scheduler)** is a more modern algorithm based on the DSR (Difficulty, Stability, Retrievability) model [6]. It considers three variables:
- **Stability** ($S$) — storage strength; higher = slower forgetting
- **Retrievability** ($R$) — retrieval strength; lower = higher forgetting probability
- **Difficulty** ($D$) — inherent complexity of the material

The key memory laws modeled: (1) more complex material → lower stability increase, (2) higher stability → lower stability increase (stabilization decay), (3) lower retrievability → higher stability increase (stabilization curve) [6]. FSRS supports a configurable `desired_retention` parameter (default 0.9) that schedules reviews at the point where predicted recall probability drops to that threshold [7]. An optional optimizer can compute personalized parameters from review history [7].

The `py-fsrs` package (MIT, 444 GitHub stars, Python 3.10+) provides a clean API:
```python
from fsrs import Scheduler, Card, Rating
scheduler = Scheduler(desired_retention=0.9)
card = Card()  # due immediately
card, review_log = scheduler.review_card(card, Rating.Good)
# card.due = next review datetime
```
It uses 21 model weights, supports learning/relearning steps, JSON serialization, and an optimizer that computes personalized parameters from `ReviewLog` history [7].

#### 1.3 Interleaved Practice

Interleaved practice — alternating between different material within a single session — is closely related to spaced repetition but operates on a shorter timescale. Piano Practice Assistant implements this by choosing a large section and practicing its subsections at random, switching every three minutes, then revisiting those subsections later in the session [3]. Modacity recommends a concrete interleaving pattern for three passages in 30 minutes: A(4min) → B(3min) → A(3min) → C(4min) → B(5min) → A(3min) → C(6min) → B(2min) [8].

#### 1.4 Application to PickHero

PickHero's `SongRecord` dataclass (`progress.py:11-24`) already tracks `attempts`, `last_played`, `section_history`, and `tempo_history` — the raw data needed for spaced repetition scheduling. The current `recommendations.py` engine is purely reactive (compares last attempt to recent average). It has no concept of:
- **Forgetting curves** — when will the player's accuracy on this song decay?
- **Due-for-review scheduling** — which song should they practice *today*?
- **Section-level spacing** — which specific bars need revisit?

The `ProgressTracker.record_detailed_result()` method already appends to `section_history` and `tempo_history` with per-attempt detail. Adding an FSRS or SM-2 scheduler that treats each song (or section) as a "card" and uses accuracy as the review rating would be a natural extension. The `last_played` timestamp is already stored; computing the interval since last practice is trivial.

### 2. Commercial Practice App Architectures

#### 2.1 Yousician — Personal Learning Path + Gamification

Yousician structures practice around a **personal learning path** with thousands of lessons, exercises, and videos, organized by skill level [9]. Key architectural elements:

- **Instant feedback on accuracy and timing** — the app listens to the player and gives real-time feedback on precision and timing [9]
- **Gamified progression** — "Earn rewards, beat high scores, and level up as you learn new skills" [9]
- **Lesson plans by real music teachers** — structured curriculum rather than free-form song practice [9]
- **Song library integration** — 2000+ songs spanning genres, used as practice material within the learning path [9]
- **Mission/Quest structure** — the learning path is broken into missions, each teaching a specific technique or set of chords, with songs as the "boss battles" that test the accumulated skills

The app averages 17-minute sessions, suggesting the design is optimized for short, focused practice bursts rather than marathon sessions [9].

#### 2.2 Rocksmith+ — Riff Repeater & Adaptive Difficulty

Rocksmith+ (the subscription successor to Rocksmith 2014) provides several practice tools directly relevant to PickHero [10]:

- **Riff Repeater** — section loop tool that isolates a passage, slows it down, and repeats it. The player can set loop boundaries, adjust tempo independently of the song, and practice the section until mastered. This is the commercial equivalent of PickHero's `L` key loop-weakest-section.
- **Adaptive difficulty** — the difficulty dynamically adjusts based on the player's performance. Notes are gradually added or removed from the arrangement as the player demonstrates proficiency. This is a key UX pattern PickHero lacks: the *difficulty itself* adapts, not just the tempo.
- **Speed-up settings** — customizable tempo progression: the player configures how quickly the tempo increases during practice (e.g., +5 BPM after each successful loop, or hold tempo until accuracy threshold met). The official help article on this exists but returned empty content when scraped [11].
- **Real-time evaluation** — "Rocksmith+ continuously records and analyzes your performance and provides immediate recommendations to help you learn efficiently" [10]
- **Video lessons + interactive sheet music** — combines video instruction with the interactive notation

#### 2.3 Simply Guitar — Structured Curriculum

Simply Guitar's website was minimally scrapeable (returned only a tuner widget), but its known architecture from secondary sources includes: a linear curriculum path from absolute basics (string names, first chords) through intermediate techniques, with each lesson building on the previous. The app uses the phone's microphone for pitch detection and provides real-time feedback similar to Yousician but with a more traditional lesson structure [12].

#### 2.4 Logato — Algorithmic Practice Session Generation

Logato is the most architecturally relevant commercial tool for PickHero because it explicitly applies spaced repetition to music practice [2]. Its session generation algorithm:

1. **Input**: available time + optional focus areas
2. **Priority scoring**: each exercise in the library gets a score based on due-date decay, BPM gap, fail count, discovery score, focus-area relevance, and goal urgency
3. **Session structure**: warmup → core (technique) → core (ear training) → core (theory) → cooldown, with time allocated proportionally to each pillar
4. **In-session metronome** with BPM tracking
5. **Multi-metric self-assessment**: Technical Accuracy, Rhythm, Comfort, Overall Satisfaction — the algorithm weighs all four [2]

This is the model for what PickHero's `recommendations.py` could become: not just post-run text hints, but a pre-session planner that selects *what* to practice and *in what order*.

#### 2.5 Key UX Patterns from Commercial Apps

| Pattern | Yousician | Rocksmith+ | Logato | PickHero Status |
|---------|-----------|------------|--------|-----------------|
| Personal learning path | ✅ Missions/quests | ❌ Song-based | ✅ Session generation | ❌ No curriculum |
| Section loop practice | ✅ | ✅ Riff Repeater | ❌ | ✅ `L` key (basic) |
| Adaptive difficulty | ✅ | ✅ Notes add/remove | ❌ | ❌ Fixed difficulty |
| Tempo auto-progression | ✅ | ✅ Speed-up settings | ❌ | ❌ Manual tempo only |
| Real-time timing feedback | ✅ Precision + timing | ✅ Real-time eval | ❌ | ✅ Timing Judge (new) |
| Spaced repetition scheduling | ❌ | ❌ | ✅ SM-2 adapted | ❌ |
| Gamified progression | ✅ XP, rewards, levels | ✅ | ✅ Streaks, mastery | ❌ Text recommendations only |
| Post-run analytics | ✅ | ✅ | ✅ Dashboard | ✅ Timing histogram (new) |

### 3. BPM Training Protocols — The Science of Tempo Progression

#### 3.1 The Speed-Accuracy Tradeoff

Motor learning research consistently shows a speed-accuracy tradeoff: as tempo increases, accuracy decreases, but the relationship is not linear. There is typically a "cliff zone" where accuracy drops sharply. ClefArc's guide documents this for guitar scales: the 110-130 BPM range is where "accuracy >85% drops to 62%" [4]. This cliff varies by technique (harmonic minor often ceilings at 115-125 BPM) and by individual.

PickHero's current tempo recommendation logic (`recommendations.py:130-148`) uses fixed thresholds: suggest +5% above 90% accuracy, suggest -5% below 60% accuracy. This is reasonable but misses the cliff-zone phenomenon. A better approach would track the *accuracy-vs-tempo curve* per song and detect the cliff dynamically.

#### 3.2 Tempo Progression Protocols

Multiple sources converge on similar tempo progression protocols:

**The 10-BPM Ladder** (FinalGuitar [13], ClefArc [4]):
- Start at 60 BPM with quarter notes
- Increase in 10 BPM increments (60→70→80→...)
- When reaching the limit, return to 60 BPM with eighth notes (double the note density)
- Continue with 16th notes at progressively slower tempos

**Progressive Speed Bursts** (FinalGuitar [13]):
- Start at 50% of max speed
- Play a challenging passage for 8 bars
- Increase tempo by 5 BPM
- Continue until reaching the limit
- "This pushes your technical boundaries systematically" [13]

**The 10-15-10 Method** (ClefArc [4]):
- 10 seconds slow motion (60% speed) — encode pattern without fatigue
- 15 seconds single-syllable dictation — vocalize "ta-ta-ta" while playing to lock in muscle memory at linguistic speed
- 10 seconds game-style check — quantify errors (target <5 misplayed notes)

**Session length and retention**: Sessions under 35 minutes boost retention by 92% vs. 25% for sessions over 45 minutes [4]. This validates Yousician's 17-minute average session design [9].

#### 3.3 Fitts and Posner Three Stages of Motor Learning

[INFERENCE] The Fitts and Posner model (1967) describes three stages of motor skill acquisition:
1. **Cognitive stage** — the learner is figuring out what to do; performance is slow, error-prone, and requires conscious attention. For guitar: learning where to place fingers, which strings to pluck.
2. **Associative stage** — the learner refines the skill; errors decrease, performance becomes more consistent. For guitar: playing through a song at reduced tempo with fewer mistakes.
3. **Autonomous stage** — the skill is largely automatic; minimal conscious attention required. For guitar: playing at full speed, focusing on musicality rather than mechanics.

This framework directly informs practice design: the cognitive stage benefits from slow, accurate repetition; the associative stage benefits from tempo progression and interleaved practice; the autonomous stage benefits from performance and expression work.

PickHero's Timing Judge is most valuable in the **associative stage**, where the player has the notes but needs to refine timing. In the cognitive stage, the timing feedback may be premature (the player is still figuring out *what* to play, not *when*). A practice mode that adapts feedback type to the learner's stage would be ideal.

#### 3.4 Application to PickHero

PickHero's `_tempo_recommendation()` function (`recommendations.py:130-148`) uses percentage-based tempo factors (0.5-1.0). The plan's `tempo_factor` represents the fraction of full speed. Current logic:
- At full speed (1.0): suggest slower if accuracy < 60%, praise if ≥ 90%
- Below full speed: suggest +5% if ≥ 90%, suggest -5% if < 60%

Improvements informed by research:
1. **Track the accuracy-vs-tempo curve** per song across attempts. Store this in `tempo_history` (already exists in `SongRecord`). Detect the cliff zone dynamically rather than using fixed thresholds.
2. **Use BPM-based increments** (5-10 BPM) rather than percentage-based (5%). At 120 BPM, 5% = 6 BPM; at 60 BPM, 5% = 3 BPM. BPM-based increments are more consistent with motor learning research.
3. **Add progressive speed burst mode** — a guided practice mode that loops a section and auto-increases BPM after successful runs (accuracy ≥ threshold for N consecutive loops).
4. **Session time awareness** — the 35-minute retention cliff suggests PickHero should encourage shorter, focused sessions. A simple session timer with a "take a break" reminder after 35 minutes would help.

### 4. Rhythm Training Beyond Hit/Miss

#### 4.1 Micro-timing Deviations (MTDs)

The Nature study on swing feel in jazz provides the most rigorous framework for understanding micro-timing in music [14]. Key findings:

- **MTDs are defined as minute timing deviations from strict metronomic regularity**, typically 10-50ms [15]
- **Perception thresholds**: Trained musicians can detect deviations as small as 2.5% of beat length (12.5ms at 120 BPM); non-musicians at 4.4% (22ms at 120 BPM) [15]
- **The groove paradox**: Quantized (perfectly timed) patterns are often rated *more* groovy than patterns with MTDs. Increasing MTD magnitude generally decreases perceived groove quality. However, *very small* deviations (1-2%) can improve groove [14][15]
- **Early shifts are rated more negatively than late shifts** [15]
- **MTD patterns are genre-specific**: jazz has larger MTDs than rock; the pattern (not just magnitude) matters [14]

PickHero's Timing Judge already captures the raw timing error per onset (`TimingObservation.timing_error_ms` in `timing.py`). The ±25ms ON_TIME threshold and ±100ms histogram are well-grounded in the perception thresholds above.

#### 4.2 Swing Ratio Analysis

The swing ratio — the length ratio of consecutive 8th notes in the long-short pattern — is a well-studied metric in jazz [14]. The typical swing ratio varies with tempo: at slower tempos, musicians use more pronounced swing (ratio ~2:1, the triplet feel); at faster tempos, the swing ratio approaches 1:1 (straight 8ths) [14].

Roger Linn's MPC swing implementation is the industry standard [15]:
- **50% swing** = straight (1:1 ratio)
- **54%** = slight push, loosens feel
- **58%** = light swing
- **62%** = medium swing
- **66%** = perfect triplet (2:1)
- **70%** = heavy swing

The algorithm delays every even-numbered 16th note within each 8th note pair [15].

**Application to PickHero**: If the song has 8th-note passages, PickHero could compute the player's effective swing ratio by measuring the timing between consecutive 8th notes and comparing to the theoretical straight timing. This would reveal whether the player is rushing the off-beats (swing ratio < expected) or laying back (swing ratio > expected). The implementation would use the existing `TimingObservation` data — group consecutive 8th-note observations by measure, compute the average inter-onset interval ratio, and compare to the song's expected swing ratio (from the GP tab's tempo/time-signature metadata).

#### 4.3 Rushing vs. Dragging — Timing Trend Analysis

The Nature study's MTD manipulation includes "inverting" timing deviations: "playing ahead of the beat ('dragging') becomes playing before the beat ('rushing'), and vice-versa" [14]. (Note: the paper's terminology appears to swap the conventional meanings — in standard musician usage, "rushing" = playing ahead of the beat, "dragging" = playing behind.)

PickHero's `TimingStats` already computes `mean_error_ms`, which indicates the overall rushing (negative) or dragging (positive) tendency. But it doesn't track *trends* over the course of the song. A player who starts on-time but drifts late as the song progresses is exhibiting fatigue; a player who is consistently early is rushing. Computing timing error as a function of song position (early/mid/late thirds) would reveal these patterns.

The `per_measure` dict in `TimingStats` (`timing.py`) already buckets by measure. A simple addition: compute the slope of `mean_error_ms` across measures (linear regression of error vs. measure index). A significantly negative slope = player is improving (getting more on-time); positive slope = player is fatiguing or losing focus.

#### 4.4 Participatory Discrepancies and Groove

Charles Keil's theory of participatory discrepancies (PD theory) claims that "the little discrepancies within a jazz drummer's beat, between bass and drums, between rhythm section and soloists, that create 'swing' and invite us to participate" [14]. While empirical studies have had mixed results (quantized patterns often rated as groovier), the concept is relevant for practice: a player learning to control their timing deviations — not eliminate them, but *intentionally* shape them — is developing musical expression.

**Application to PickHero**: The Timing Judge could add a "groove mode" that compares the player's timing deviations to a reference recording's deviations (if available from the MIDI backing track or a pre-analyzed reference). This is ambitious but would be a unique feature. More practically: computing the *consistency* of timing deviations ($\sigma$) and rewarding low $\sigma$ (consistent feel) rather than just low mean error (mechanical accuracy).

#### 4.5 What the Codebase Already Does Well

PickHero's `timing.py` already implements:
- Per-onset `TimingObservation` with `timing_error_ms` and `TimingVerdict` (EARLY/ON_TIME/LATE/MISSED/EXTRA) [verified: `timing.py:25-48`]
- `TimingStats` with mean error, std dev, early/late/on-time/missed/extra counts, min/max error, per-measure buckets, and 20-bin histogram covering ±100ms [verified: `timing.py:52-75`]
- `PitchVerdict` enum (CORRECT/NEAR/WRONG/UNKNOWN) separating pitch from timing [verified: `timing.py:35-40`]
- The scrolling UI already renders the histogram with colored bars and worst-bars listing [verified: `scrolling.py:1099-1143`]

What it lacks (informed by the research):
- Swing-ratio computation for 8th-note passages
- Timing-trend analysis (slope of error across song position)
- Groove consistency metric ($\sigma$ of deviations in a different way than overall $\sigma$)
- Comparison to a reference timing profile

### 5. Open-Source Practice Tools

#### 5.1 TuxGuitar

TuxGuitar is an open-source multitrack tablature editor and player written in Java (LGPL) [16][17]. It is the direct open-source competitor to Guitar Pro. Key features relevant to PickHero:

- **Multi-format import**: opens Guitar Pro (GP3-GP5), Power Tab Editor, TablEdit, and MIDI files — same formats PickHero handles via pyguitarpro [16]
- **Playback with tempo control**: standard tab player with playback speed adjustment
- **No practice intelligence**: TuxGuitar is a tab editor/player, not a practice tool. It has no section looping, no timing feedback, no practice recommendations, no progress tracking
- **SWT and JavaFX versions**: two UI backends, suggesting the project values UI flexibility [16]

TuxGuitar represents the "tab player" baseline — what PickHero was before adding its practice features. PickHero's differentiation is the real-time audio detection and timing feedback layer on top of tab playback.

#### 5.2 Solfej

Solfej is a free web app providing "chord & scale diagrams, music theory lessons, and ear training tools for guitar and piano players" [18]. Its metadata describes it as focused on theory and ear training rather than real-time performance feedback. Key features:
- Chord and scale diagrams (visual fretboard reference)
- Music theory lessons (structured curriculum)
- Ear training exercises (interval recognition, chord identification)
- Built with Gatsby (static site generator), suggesting a lightweight web-first approach

Solfej is relevant as a reference for **theory education integration** — a feature PickHero lacks entirely. If PickHero wanted to add theory context (e.g., "you're playing a pentatonic scale here" or "this section uses a ii-V-I progression"), Solfej's approach of visual diagrams + structured lessons would be the model.

#### 5.3 GNU Solfege

GNU Solfege is free ear-training software written in Python (GPL v3+) [19]. It is designed to be "easily extended with lessonfiles (data files), so the user can create new exercises" [19]. Key characteristics:
- **Exercise-driven**: exercises are defined in data files, making the system extensible
- **Python-based**: same language as PickHero, suggesting potential code/architecture sharing
- **GTK-based UI**: uses PyGTK/Gtk+ (legacy) or Gtk+ 3 (development branch) [19]
- **Focus on ear training**: interval recognition, chord identification, rhythm dictation, sight-reading
- **No real-time audio analysis**: exercises are mouse/keyboard based, not microphone-based
- **Production status**: stable (v3.22.2), maintained since 2002 [19]

GNU Solfege's exercise-file architecture is relevant: it separates exercise *content* from exercise *logic*. PickHero could adopt a similar pattern for practice exercises — defining practice routines (warmup scales, chord transitions, timing drills) as data files rather than hardcoding them.

#### 5.4 PulseKeeper

PulseKeeper is a free web-based rhythm timing trainer with features directly relevant to PickHero's timing goals [20]:
- **Tap-based timing training**: tap in time with a metronome using a button, spacebar, or MIDI controller
- **Visual feedback**: green = perfect, yellow = close, red = keep practicing
- **Subdivision practice**: quarter notes, 8th notes, triplets, 16th notes
- **Difficulty levels**: Easy/Medium/Hard/Expert (tighter timing windows)
- **Auto-calibration**: measures system audio latency using the microphone [20]
- **Audio input detection**: supports drum/instrument input via microphone
- **Practice plans**: Free Play, # of Taps, or Time-based sessions

PulseKeeper's difficulty-level system (tighter timing windows at higher difficulty) is a simple but effective pattern PickHero could adopt: instead of a binary Timing Judge on/off, offer "Timing Judge: Relaxed (±50ms) / Standard (±25ms) / Strict (±10ms) / Expert (±5ms)".

### 6. Concrete Recommendations for PickHero

#### 6.1 Spaced Repetition Scheduler (Highest Impact)

**What**: Add a `PracticeScheduler` class that uses SM-2 or FSRS to schedule song/section review.

**How**: Treat each song as an FSRS "card." After each play, use accuracy as the review rating:
- accuracy ≥ 95% → `Rating.Easy`
- accuracy 85-94% → `Rating.Good`
- accuracy 70-84% → `Rating.Hard`
- accuracy < 70% → `Rating.Again`

The scheduler computes the next due date. The main menu surfaces "Due for review: N songs" and sorts the song list by urgency.

**Where**: New file `pickhero/scheduler.py`. Integrate with existing `ProgressTracker` — the scheduler reads `SongRecord.attempts`, `SongRecord.last_played`, and `SongRecord.best_accuracy` to compute intervals. Use `py-fsrs` (MIT, pip installable) or implement SM-2 inline (~30 lines).

**Data**: The existing `progress.json` already stores per-song records. Add `next_review_due: str` and `fsrs_state: dict` fields to `SongRecord`. The scheduler updates these in `record_detailed_result()`.

#### 6.2 Adaptive Tempo Progression with Cliff Detection

**What**: Replace the fixed 5% tempo increment with an adaptive system that tracks the accuracy-vs-tempo curve per song and detects the cliff zone.

**How**: In `recommendations.py`, add a `_tempo_curve_analysis()` function that examines `record.tempo_history` to find the tempo at which accuracy drops sharply. Store the detected cliff as `record.cliff_bpm`. The tempo recommendation then suggests practicing *just below* the cliff (building consistency) before attempting to break through it.

Use BPM-based increments: 10 BPM below 100 BPM, 5 BPM above. This aligns with the 10-BPM ladder protocol from FinalGuitar [13].

**Where**: `recommendations.py` — extend `_tempo_recommendation()`. Add `cliff_bpm: float | None` to `SongRecord` in `progress.py`.

#### 6.3 Guided Practice Loop with Auto-Progression

**What**: A structured practice mode that auto-loops the weakest section at reduced tempo and auto-increases tempo when the player hits an accuracy threshold for N consecutive successful loops.

**How**: New state in `PlayingScreen` (`scrolling.py`): `self._guided_practice: bool = False`, `self._guided_loop_count: int = 0`, `self._guided_target_accuracy: float = 90.0`, `self._guided_consecutive_successes: int = 0`. When enabled:
1. Auto-seek to the weakest section start (from `self._weakest_sections[0]`)
2. Set tempo to `cliff_bpm - 10` or `50% of full speed` if no cliff detected
3. After each loop, check section accuracy. If ≥ target for 3 consecutive loops, increase tempo by 5 BPM. If < 60%, decrease by 5 BPM (floor at 50%).
4. Display a "Guided Practice" HUD with: current tempo, consecutive successes, target tempo, and a progress bar.

**Where**: `scrolling.py` — new `_guided_practice_update()` method called from `update()`. New key binding (e.g., `G`) to toggle guided practice. The existing `L` key loop logic (`scrolling.py:410-416`) provides the seek-back mechanism.

#### 6.4 Timing Trend Analysis

**What**: Add a timing-trend metric to `TimingStats` that detects rushing/dragging/fatiguing patterns across the song.

**How**: In `timing.py`, add to `TimingStats`:
```python
timing_slope_ms_per_measure: float = 0.0  # linear regression slope
trend: str = "stable"  # "rushing", "dragging", "fatiguing", "improving", "stable"
```
Compute in `compute_stats()`: regress `timing_error_ms` against `measure` index across all observations with non-NaN errors. Classify:
- slope < -0.5 → "improving" (getting more on-time)
- slope > 0.5 → "fatiguing" (drifting later)
- mean < -10 → "rushing" (consistently early)
- mean > 10 → "dragging" (consistently late)
- else → "stable"

Display this in the timing summary screen alongside the histogram.

**Where**: `timing.py` — extend `TimingStats` and `compute_stats()`. `scrolling.py` — extend `_draw_timing_summary()` to render the trend.

#### 6.5 Swing Ratio Detection (Medium Priority)

**What**: For songs with 8th-note passages, compute the player's effective swing ratio and compare to the expected ratio.

**How**: In `timing.py`, add a `SwingAnalysis` dataclass:
```python
@dataclass
class SwingAnalysis:
    measured_ratio: float  # average long/short ratio
    expected_ratio: float  # 1.0 for straight, ~1.5-2.0 for swing
    swing_percentage: float  # (ratio-1)/(2-1)*100 + 50, MPC-style
    consistency: float  # std dev of ratio
```
Compute by grouping consecutive 8th-note `TimingObservation`s by measure, measuring the inter-onset interval ratio between long-short pairs. This requires the tab's note duration metadata (available from pyguitarpro via the timeline).

**Where**: `timing.py` — new `SwingAnalysis` and `compute_swing()`. `matcher.py` — call `compute_swing()` when timing judge is enabled and the song has 8th-note passages. `scrolling.py` — display in timing summary.

#### 6.6 Session Timer with Retention Awareness (Low Priority)

**What**: A session timer that reminds the player to take a break after 35 minutes, based on the retention cliff research [4].

**How**: Track `self._session_start_time` in `PlayingScreen`. After 35 minutes of active practice (not menu time), show a gentle "Take a 5-minute break — your retention drops after 35 minutes" overlay. This is a 15-line addition.

**Where**: `scrolling.py` — add to `update()` or a dedicated `_check_session_time()`.

### 7. Priority Assessment

| Improvement | Impact | Effort | Research Backing | PickHero Gap |
|-------------|--------|--------|-----------------|--------------|
| Spaced repetition scheduler | Very High | Medium | $d=0.42$ across 116 studies [1]; SM-2/FSRS proven [6][7]; Logato validates for music [2] | No due-for-review scheduling |
| Adaptive tempo + cliff detection | High | Medium | Cliff zones documented [4]; 10-BPM ladder protocol [13] | Fixed 5% increments, no curve tracking |
| Guided practice loop | High | Medium | Rocksmith Riff Repeater [10]; Yousician mission path [9] | Basic `L` loop, no auto-progression |
| Timing trend analysis | Medium | Low | MTD research [14][15]; rushing/dragging detection | Mean error only, no trend |
| Swing ratio detection | Medium | High | Swing ratio research [14]; MPC standard [15] | No swing analysis |
| Session timer | Low | Low | 35-min retention cliff [4] | No session concept |

## Sources

1. Donovan & Radosevich (1999), "A meta-analytic review of the distribution of practice effect" — cited in [3] and [5]. Effect size $d=0.42$ across 116 studies since 1927.
2. Logato — Practice Intelligence for Musicians — https://www.logatomusic.com/ (retrieved 2026-07-03). SM-2 adapted for music; priority scoring with due-date decay, BPM gap boost, fail count, discovery score, focus-area relevance, goal urgency.
3. Piano Practice Assistant — "Spaced repetition for musicians" — http://pianopracticeassistant.com/spaced-repetition/ (retrieved 2026-07-03). Discusses spacing effect for piano, massed vs. spaced for motor skills, interleaved practice within sessions.
4. ClefArc — "How to Practice Guitar Scales Effectively for Speed and Accuracy" — https://clefarc.com/blogs/guitar-performance-technique/how-to-practice-guitar-scales-effectively-for-speed-and-accuracy-5-key-phases-for-mastery (retrieved 2026-07-03). Documents cliff zone at 110-130 BPM, 10-15-10 method, 35-min retention cliff, tempo ladder protocol.
5. Gwern Branwen — "Spaced Repetition for Efficient Learning" — https://gwern.net/spaced-repetition (retrieved 2026-07-03). Comprehensive review of spacing effect literature, forgetting curves, testing effect.
6. FSRS — Free Spaced Repetition Scheduler — https://github.com/open-spaced-repetition/free-spaced-repetition-scheduler (retrieved 2026-07-03). DSR model: Difficulty, Stability, Retrievability. Three memory laws: complexity → stability, stabilization decay, stabilization curve.
7. py-fsrs — Python package for FSRS — https://github.com/open-spaced-repetition/py-fsrs (retrieved 2026-07-03). MIT license, Python 3.10+, 444 stars. API: `Scheduler`, `Card`, `Rating`, `ReviewLog`. Supports `desired_retention`, custom parameters, optimizer, JSON serialization.
8. Modacity — "Using Spaced Repetition to Achieve Effective Practice" — https://www.modacity.co/blog/using-repetition-achieve-effective-practice/ (retrieved 2026-07-03). Recommended spacing intervals: 1 day, 7 days, 16 days, 35 days. Interleaving pattern example.
9. Yousician — Official homepage — https://yousician.com/ (retrieved 2026-07-03). Personal learning path with 9000+ lessons, instant feedback on accuracy and timing, gamified progression (rewards, high scores, levels), 17-min average session.
10. Rocksmith+ — Official site — https://rocksmith.ubisoft.com/rocksmith-2014/ (retrieved 2026-07-03). Real-time evaluation, Riff Repeater, adaptive difficulty, speed-up settings, interactive video lessons.
11. Ubisoft Help — "Customising the speed up settings in Rocksmith" — https://www.ubisoft.com/help/rocksmith-plus/gameplay/article/customising-the-speed-up-settings-in-rocksmith/000097812 (retrieved 2026-07-03). Page returned empty content when scraped; existence confirms configurable speed-up settings.
12. Simply Guitar — Official site — https://www.simplyguitar.com/ (retrieved 2026-07-03). Page returned minimal content (tuner widget only); structured curriculum architecture inferred from app store descriptions.
13. FinalGuitar — "BPM & Tempo Training Tool" — https://www.finalguitar.com/bpm-tempo-trainer/ (retrieved 2026-07-03). Progressive mode (gradual tempo increase), 10-BPM ladder exercise, progressive speed bursts (8 bars + 5 BPM), tempo memory exercise, classical tempo markings.
14. Kilchenmann, Senn, etc. — "Microtiming Deviations and Swing Feel in Jazz" — Nature Scientific Reports — https://www.nature.com/articles/s41598-019-55981-3 (retrieved 2026-07-03). Swing ratio analysis, MTD manipulation (quantize/exaggerate/invert), 160-listener survey, quantized versions preferred. References Keil's PD theory.
15. Phonon — "Microtiming Analysis Research" — https://github.com/ekg/phonon/blob/main/docs/MICROTIMING_ANALYSIS_RESEARCH.md (retrieved 2026-07-03). Perception thresholds (JND 2.5% trained / 4.4% non-musicians), groove paradox, optimal tempo 100-120 BPM, laid-back snare 17.4ms at 96 BPM, Roger Linn MPC swing algorithm (50%-70%), ghost note velocity guidelines.
16. TuxGuitar — GitHub — https://github.com/helge17/tuxguitar (retrieved 2026-07-03). Open-source tab editor (LGPL, Java). Multi-format import (GP, Power Tab, TablEdit, MIDI). SWT and JavaFX versions.
17. TuxGuitar — Official site — https://www.tuxguitar.app/ (retrieved 2026-07-03). Stable version 2.0.1. No practice intelligence features.
18. Solfej — Official site — https://www.solfej.io/ (retrieved 2026-07-03). Free chord & scale diagrams, music theory lessons, ear training tools. Built with Gatsby.
19. GNU Solfege — Savannah — https://savannah.gnu.org/projects/solfege/ (retrieved 2026-07-03). Free ear-training software (GPL v3+, Python). Exercise-file driven architecture. Stable v3.22.2. GTK-based UI.
20. PulseKeeper — https://pulsekeeper.net/ (retrieved 2026-07-03). Free rhythm timing trainer. Tap-based, visual feedback (green/yellow/red), subdivisions, difficulty levels, auto-calibration, audio input detection.

## Confidence & Gaps

**High confidence:**
- Spacing effect and SM-2/FSRS algorithms: well-established research with meta-analytic backing [1][5][6] and multiple implementations [7]. The adaptation to music motor learning is validated by Logato [2] and Piano Practice Assistant [3].
- Micro-timing perception thresholds: the 2.5%/4.4% JND values [15] and the ±25ms Timing Judge threshold are well-aligned.
- Tempo progression protocols (10-BPM ladder, progressive speed bursts): multiple independent sources converge on similar protocols [4][13].
- Rocksmith's Riff Repeater and adaptive difficulty: confirmed from the official site [10], though detailed mechanics (exact speed-up algorithm) could not be extracted.

**Medium confidence:**
- The 35-minute retention cliff [4]: cited from a single source (ClefArc blog) referencing "Sports Biomechanics 2023 studies." The specific study could not be directly verified. The general principle (shorter focused sessions > long unfocused ones) is well-supported but the exact 35-minute threshold should be treated as approximate.
- Fitts and Posner three-stage model [INFERENCE]: the Wikipedia article does not exist; the model is referenced in secondary sources but the primary reference (Fitts & Posner, *Human Performance*, 1967) was not directly accessed. The three stages (cognitive, associative, autonomous) are widely cited in motor learning literature.

**Low confidence / Gaps:**
- Yousician's exact mission/quest structure: the homepage [9] describes the learning path concept but specific mission mechanics (how missions are structured, what constitutes completion, how XP is awarded) could not be extracted from public pages. The app is behind a paywall.
- Simply Guitar's curriculum structure: the website returned minimal content. Architecture is inferred from app store descriptions and secondary reviews.
- Rocksmith's speed-up settings algorithm: the help article [11] exists but returned empty content. The exact auto-progression logic is unknown.
- Swing ratio computation for guitar tabs: the research [14][15] focuses on jazz piano and drums. Whether the same swing-ratio analysis applies cleanly to guitar tab practice (which may not have explicit swing notation) needs validation.
- No academic paper on *guitar-specific* practice app design was found. The commercial apps (Yousician, Rocksmith) are proprietary and do not publish their algorithms.

**Unresolved questions:**
- What is the optimal accuracy threshold for "graduating" from a tempo level? PickHero uses 90%; commercial apps don't disclose their thresholds.
- Should spaced repetition intervals be based on accuracy alone, or also on timing consistency ($\sigma$)? The FSRS model is designed for recall (binary correct/incorrect); motor skill has a continuous quality dimension that the model doesn't natively handle.
- How should PickHero handle the fact that a player's accuracy on a song is influenced by tempo? A 95% accuracy at 50% speed is very different from 95% at 100% speed. The scheduler needs to account for the tempo dimension, not just accuracy.
