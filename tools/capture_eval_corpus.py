"""Interactive terminal recorder for positive and negative corpus examples.

The input plan is itself a valid corpus JSONL manifest.  ``audio_path`` names
where each take should be written relative to the output manifest.  Prompts are
read from ``metadata.prompt`` so the plan remains machine-readable.
"""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root()

from pickhero.evaluation.manifest import load_manifest, write_manifest


def _write_wave(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a labelled guitar corpus")
    parser.add_argument("plan", type=Path, help="Capture-plan JSONL")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/evaluation/local-corpus.jsonl"),
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument(
        "--input-channel",
        type=int,
        default=1,
        help="One-based physical input channel",
    )
    parser.add_argument(
        "--lead-in",
        type=float,
        default=0.75,
        help="Silent lead-in before the displayed event start",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output manifest instead of appending",
    )
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError as exc:
        parser.error(f"sounddevice is required: {exc}")

    cases = load_manifest(args.plan)
    if not cases:
        parser.error("capture plan is empty")
    existing_ids: set[str] = set()
    if args.manifest.exists() and not args.overwrite:
        existing_ids = {case.case_id for case in load_manifest(args.manifest)}
        duplicates = sorted(existing_ids & {case.case_id for case in cases})
        if duplicates:
            parser.error(
                "output manifest already contains planned case IDs; "
                "use --overwrite or remove them: " + ", ".join(duplicates[:8])
            )
    output_cases = []
    channel_index = max(0, args.input_channel - 1)
    channels = channel_index + 1

    for position, case in enumerate(cases, start=1):
        prompt = case.metadata.get("prompt", case.case_id)
        duration = max(case.end_s + 0.25, 1.0)
        print(f"\n[{position}/{len(cases)}] {prompt}")
        print(
            f"Record {duration:.2f}s; play at {case.start_s:.2f}s. "
            "Enter=record, s=skip, q=quit"
        )
        command = input("> ").strip().lower()
        if command == "q":
            break
        if command == "s":
            continue

        if args.lead_in > 0:
            print(f"Starting in {args.lead_in:.2f}s...")
            sd.sleep(int(args.lead_in * 1000.0))
        frames = int(round(duration * args.sample_rate))
        recording = sd.rec(
            frames,
            samplerate=args.sample_rate,
            channels=channels,
            dtype="float32",
            device=args.device,
            blocking=False,
        )
        cue_s = case.expected_onset_s if case.expected_onset_s is not None else case.start_s
        sd.sleep(max(0, int(round(cue_s * 1000.0))))
        print("PLAY NOW", flush=True)
        sd.wait()
        mono = np.asarray(recording[:, channel_index], dtype=np.float32)
        audio_path = Path(case.audio_path)
        destination = (
            audio_path
            if audio_path.is_absolute()
            else args.manifest.parent / audio_path
        )
        _write_wave(destination, mono, args.sample_rate)
        output_cases.append(replace(case, audio_path=str(audio_path)))
        print(f"Saved {destination}")

    if output_cases:
        write_manifest(
            args.manifest,
            output_cases,
            append=args.manifest.exists() and not args.overwrite,
        )
        print(f"Wrote {len(output_cases)} cases to {args.manifest}")
    else:
        print("No takes recorded.")


if __name__ == "__main__":
    main()
