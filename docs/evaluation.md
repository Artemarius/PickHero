# Detector and scoring evaluation

PickHero must not claim Rocksmith-class recognition from synthetic sine waves or
from a handful of successful takes. The production verifier is measured against
a versioned JSONL corpus containing real positive performances, real mistakes,
technique-negative examples, silence/noise, and hardware/tone metadata.

## Non-negotiable rules

1. **Tune on `calibration`; report on `test`.** Never choose thresholds from the
   held-out test split.
2. **Keep every event from one recording in one split.** Otherwise adjacent
   windows from the same take leak into both calibration and test.
3. **Record real negatives.** Counterfactual expectations such as asking for F
   against audio containing E are useful smoke tests, but they do not model
   fret buzz, sympathetic strings, incomplete chords, poor bends, or accidental
   re-picks.
4. **Preserve metadata.** At minimum label guitar, pickup, interface, tone,
   tuning, and player. Aggregate F1 can hide a detector that only works on one
   clean bridge-pickup setup.
5. **Do not discard hard takes.** Mark clipping, low annotation confidence, or
   disputed performances explicitly. A failure report is more valuable than a
   cosmetically high score.

## Manifest schema

Each line is an independent JSON object. See
`research/evaluation/capture-plan.example.jsonl` for complete examples.

Important fields:

- `split`: `calibration`, `test`, or `development`.
- `event_kind`: `single_note`, `chord`, `technique`, or `silence`.
- `expected_present`: whether the requested pitch/chord is actually present.
- `technique_present`: independent articulation label for technique cases.
- `negative_reason`: required for negative pitch/chord/silence cases.
- `notes`: expected MIDI/string/fret/role records.
- `metadata`: guitar, pickup, interface, tone, tuning, player, and any useful
  recording conditions.
- `technique_context`: bend target, slide endpoints, authored curve, and other
  verifier-specific ground truth.

## Capture local performances

Copy and expand the example plan. Replace metadata placeholders and add many
players, guitars, interfaces, gain structures, tunings, frets, and intentional
mistakes.

```bash
python tools/capture_eval_corpus.py \
  research/evaluation/capture-plan.example.jsonl \
  --manifest research/evaluation/local-corpus.jsonl \
  --device 3 --input-channel 2
```

The recorder writes mono PCM WAV files and a manifest whose paths remain
relative to the manifest directory.

## Import public datasets

Dataset importers normalize GOAT, GuitarSet, Guitar-TECHS, and IDMT into the
existing cache. Convert that cache to the stricter evaluator manifest:

```bash
python tools/build_eval_manifest.py \
  --cache-dir ~/.pickhero/datasets \
  --output research/evaluation/public-corpus.jsonl \
  --split-group player
```

`--counterfactual-negatives` is intentionally opt-in and every generated case
is tagged `provenance=counterfactual`.

## Validate the corpus before scoring

```bash
python tools/validate_eval_corpus.py research/evaluation/local-corpus.jsonl \
  --group-key audio_path --group-key player
```

The validator detects missing files, duplicate split groups, missing hardware
metadata, a lack of recorded negative cases, and techniques that only have one
class. For an unrecorded capture plan, pass `--allow-missing-audio`.

## Evaluate without leakage

Run calibration and test into separate directories:

```bash
python tools/evaluate_corpus.py research/evaluation/local-corpus.jsonl \
  --split calibration --mode judge --output evaluation-results/calibration

python tools/evaluate_corpus.py research/evaluation/local-corpus.jsonl \
  --split test --mode judge --output evaluation-results/test
```

Each run produces:

- `records.jsonl`: all scalar scores and detector observations.
- `failures.jsonl`: false accepts, false rejects, onset failures, and technique
  misclassifications.
- `summary.json`: overall and per-source/tone/interface/etc. metrics.
- `report.md`: compact human-readable failure report.

The summary records the exact verification policy, evaluator configuration,
PickHero version, and manifest SHA-256.

## Calibrate thresholds

Only feed records from a calibration run:

```bash
python tools/calibrate_thresholds.py \
  evaluation-results/calibration/records.jsonl \
  --max-false-accept-rate 0.01
```

The command searches scalar thresholds independently for notes, chords, and
techniques. It does not rewrite runtime constants automatically; proposed
changes still need review against the failure clips and a fresh held-out run.

## Reject detector regressions

Compare a candidate run against the last accepted held-out baseline. The
comparison checks overall results and every sufficiently large source, tone,
interface, guitar, pickup, tuning, event-kind, and technique slice.

```bash
python tools/compare_eval_runs.py \
  evaluation-results/baseline/summary.json \
  evaluation-results/candidate/summary.json
```

## Enforce release gates

`research/evaluation/quality-gates.json` contains aspirational Rocksmith-class
acceptance targets, including minimum corpus size. They are intentionally much
stricter than a development smoke test.

```bash
python tools/check_quality_gates.py \
  evaluation-results/test/summary.json
```

A detector change is accepted only when it improves the target failure mode and
does not materially regress another tone, instrument, interface, tuning, skill
level, or technique slice.

## Required corpus coverage

A serious held-out corpus should include at least:

- Every guitar string across open notes, low/middle/high frets, and common
  alternate tunings.
- Clean, crunch, high gain, compression, chorus/modulation, microphone input,
  and realistic room/interface noise.
- Single coils, humbuckers, active pickups, bass, acoustic guitar, and several
  inexpensive USB cables/interfaces.
- Correct notes plus adjacent-fret mistakes, octave aliases, sympathetic
  resonance, fret buzz, clipped takes, weak attacks, early releases, and
  accidental extra strings.
- Open/closed/inverted chords, omitted/doubled tones, sus/add/seventh families,
  partial strums, and foreign notes.
- Positive and negative bends, releases, slides, vibrato, hammer-ons,
  pull-offs, palm mutes, harmonics, and dead notes.
- Multiple players from beginner to advanced. Player identity must be grouped
  when constructing splits so the test set measures generalization rather than
  memorization of one player's attack and vibrato.

## Evaluate the multi-resolution front end

The production `CompositeVerifier` now fuses the fixed spectral verifier with
multi-resolution fundamental hypotheses for single notes and uses the latter for
chords. Evaluation runs should therefore retain at least these failure slices:

- octave and subharmonic aliases;
- E1/E2 and alternate-tuning low fundamentals;
- high-gain spectra with strong upper harmonics;
- foreign pitch classes versus legitimate chord overtones;
- worker-overrun recordings, which must be excluded from detector-threshold
  calibration and reported as capture failures instead.

Latency claims should identify whether the profile was manually trimmed or
measured with the automatic output-to-input probe. Low-confidence acoustic
measurements must not be accepted as ground truth.
