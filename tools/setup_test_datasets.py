"""Document and optionally fetch small test subsets for the dataset importers.

This script does NOT download multi-GB corpora automatically. It prints the
official URLs and license notes, then either:

1. Validates existing dataset directories you already downloaded, or
2. Generates tiny synthetic sample files in ``tests/fixtures/datasets/`` so the
   importers have something to parse in CI.

Usage:
    # Show official download URLs and validate existing paths
    python tools/setup_test_datasets.py

    # Generate CI-sized synthetic fixtures (requires the [datasets] extra)
    pip install -e .[datasets]
    python tools/setup_test_datasets.py --fixtures

    # Point at real downloaded datasets and verify importers (requires [datasets])
    python tools/setup_test_datasets.py \
        --goat ~/datasets/GOAT \
        --guitarset ~/datasets/GuitarSet \
        --guitar-techs ~/datasets/Guitar-TECHS \
        --idmt ~/datasets/IDMT-SMT-Guitar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_root()

from pickhero.datasets import DatasetRegistry


DATASET_INFO = {
    "GOAT": {
        "url": "https://github.com/JackJamesLoth/GOAT-Dataset",
        "license": "Check repository license before use",
        "note": "Guitar On Audio and Tablatures: paired tablature + audio.",
    },
    "GuitarSet": {
        "url": "https://github.com/marl/GuitarSet",
        "license": "CC BY-NC-SA 4.0 (verify on release page)",
        "note": "JAMS annotations + hexaphonic audio.",
    },
    "Guitar-TECHS": {
        "url": "https://guitar-techs.github.io/",
        "license": "Check current release terms on project page",
        "note": "Electric guitar techniques with MIDI annotations.",
    },
    "IDMT": {
        "url": "https://zenodo.org/records/7544110",
        "license": "Free for research / non-commercial (Fraunhofer IDMT)",
        "note": "IDMT-SMT-Guitar automatic transcription database.",
    },
}


def _print_urls() -> None:
    print("Official dataset URLs and license notes:\n")
    for name, info in DATASET_INFO.items():
        print(f"{name}")
        print(f"  URL:     {info['url']}")
        print(f"  License: {info['license']}")
        print(f"  Note:    {info['note']}")
        print()


def _write_jams_event(jams_path: Path, midi: int, start: float, duration: float) -> None:
    """Write a minimal JAMS note_midi annotation."""
    data = {
        "file_metadata": {
            "title": jams_path.stem,
            "artist": "",
            "release": "",
            "duration": start + duration + 0.1,
            "identifiers": {},
            "jams_version": "0.3.1",
        },
        "annotations": [
            {
                "namespace": "note_midi",
                "data": [
                    {
                        "time": start,
                        "duration": duration,
                        "value": float(midi),
                        "confidence": None,
                    }
                ],
                "annotation_metadata": {},
                "sandbox": {},
            }
        ],
        "sandbox": {},
    }
    jams_path.write_text(json.dumps(data, indent=2))


def _write_minimal_midi(midi_path: Path) -> None:
    """Write a one-note type-0 MIDI file using mido if available."""
    import mido

    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="e", time=0))
    track.append(mido.Message("note_on", note=64, velocity=100, time=0))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(str(midi_path))


def _generate_fixtures(fixtures_dir: Path) -> None:
    """Create tiny synthetic files the importers can parse."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # GOAT: JSONL + fake WAV header.
    goat_dir = fixtures_dir / "GOAT"
    goat_dir.mkdir(exist_ok=True)
    audio = goat_dir / "sample.wav"
    audio.write_bytes(b"RIFF" + (b"\x00" * 100))
    jsonl = goat_dir / "sample.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "midi": 64,
                "start": 0.0,
                "end": 0.2,
                "string": 2,
                "fret": 5,
                "technique": "normal",
                "confidence": 1.0,
            }
        )
        + "\n"
    )

    # GuitarSet: JAMS + fake WAV.
    gs_dir = fixtures_dir / "GuitarSet"
    gs_dir.mkdir(exist_ok=True)
    (gs_dir / "sample.wav").write_bytes(b"RIFF" + (b"\x00" * 100))
    _write_jams_event(gs_dir / "sample.jams", midi=60, start=0.0, duration=0.2)

    # Guitar-TECHS: real minimal MIDI under the expected folder layout.
    gt_dir = fixtures_dir / "Guitar-TECHS" / "P1_singlenotes"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "audio" / "micamp").mkdir(parents=True, exist_ok=True)
    (gt_dir / "midi").mkdir(parents=True, exist_ok=True)
    audio = gt_dir / "audio" / "micamp" / "micamp_sample.wav"
    audio.write_bytes(b"RIFF" + (b"\x00" * 100))
    midi = gt_dir / "midi" / "midi_sample.mid"
    _write_minimal_midi(midi)

    # IDMT: XML + fake WAV, mirroring the real layout.
    idmt_dir = fixtures_dir / "IDMT"
    idmt_dir.mkdir(exist_ok=True)
    (idmt_dir / "audio").mkdir(exist_ok=True)
    (idmt_dir / "annotation").mkdir(exist_ok=True)
    audio = idmt_dir / "audio" / "sample.wav"
    audio.write_bytes(b"RIFF" + (b"\x00" * 100))
    (idmt_dir / "annotation" / "sample.xml").write_text(
        "<instrumentRecording>\n"
        "  <globalParameter>\n"
        "    <audioFileName>sample.wav</audioFileName>\n"
        "  </globalParameter>\n"
        "  <transcription>\n"
        "    <event>\n"
        "      <pitch>67</pitch>\n"
        "      <onsetSec>0.0</onsetSec>\n"
        "      <offsetSec>0.2</offsetSec>\n"
        "      <fretNumber>12</fretNumber>\n"
        "      <stringNumber>1</stringNumber>\n"
        "      <excitationStyle>PK</excitationStyle>\n"
        "      <expressionStyle>NO</expressionStyle>\n"
        "    </event>\n"
        "  </transcription>\n"
        "</instrumentRecording>\n"
    )

    print(f"Synthetic fixtures written to {fixtures_dir}")


def _validate_paths(paths: dict[str, Path]) -> None:
    """Run importers on the provided directories and report event counts."""
    registry = DatasetRegistry(
        dataset_paths={k: str(v) for k, v in paths.items()},
    )
    events = registry.scan_datasets()
    print()
    for source in paths:
        count = sum(1 for e in events if e.source == source)
        print(f"  {source}: {count} event(s)")
    print(f"Total: {len(events)} event(s) cached to {registry._cache_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or generate dataset fixtures for PickHero."
    )
    parser.add_argument("--fixtures", action="store_true", help="Generate CI fixtures.")
    parser.add_argument("--goat", type=Path, help="Path to GOAT dataset.")
    parser.add_argument("--guitarset", type=Path, help="Path to GuitarSet dataset.")
    parser.add_argument(
        "--guitar-techs", type=Path, help="Path to Guitar-TECHS dataset."
    )
    parser.add_argument("--idmt", type=Path, help="Path to IDMT-SMT-Guitar dataset.")
    args = parser.parse_args()

    if args.fixtures:
        _generate_fixtures(Path("tests/fixtures/datasets"))
        return

    _print_urls()

    paths: dict[str, Path] = {}
    if args.goat:
        paths["GOAT"] = args.goat
    if args.guitarset:
        paths["GuitarSet"] = args.guitarset
    if args.guitar_techs:
        paths["Guitar-TECHS"] = args.guitar_techs
    if args.idmt:
        paths["IDMT"] = args.idmt

    if paths:
        _validate_paths(paths)
    else:
        print("or pass --goat/--guitarset/--guitar-techs/--idmt to validate real data.")


if __name__ == "__main__":
    main()
