"""PickHero application entry point."""

import argparse
import sys
from pathlib import Path

from pickhero.config import Config


def _resolve_songs_dir(config: Config) -> None:
    """Fix songs_dir for frozen exe: look next to the executable."""
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path.cwd()

    songs_path = Path(config.songs_dir)
    if not songs_path.is_absolute():
        config.songs_dir = str(base_dir / songs_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pickhero", allow_abbrev=False)
    parser.add_argument("--version", action="store_true", help="Show version and exit.")
    subparsers = parser.add_subparsers(dest="command")

    console = subparsers.add_parser(
        "console",
        help="Run the audio input testing console.",
        description="Audio input testing console for pitch, chord, and synth modes.",
        allow_abbrev=False,
    )
    try:
        from pickhero.audio.console import build_console_parser
        build_console_parser(console)
    except ModuleNotFoundError as e:
        # sounddevice / aubio may be absent in a minimal install — the
        # console subcommand is optional; the main app still works.
        console.add_argument(
            "--unavailable",
            action="store_true",
            default=True,
            help=f"audio console unavailable: {e}",
        )

    return parser


def main():
    # Force X11 for stable rendering even on Wayland sessions. PyGame/SDL's
    # Wayland backend is currently unreliable here (tearing, input issues).
    import os
    if os.environ.get('WAYLAND_DISPLAY') and not os.environ.get('SDL_VIDEODRIVER'):
        os.environ['SDL_VIDEODRIVER'] = 'x11'

    parser = _build_parser()
    args = parser.parse_args()

    if args.version:
        from pickhero import __version__
        print(f"PickHero {__version__}")
        sys.exit(0)

    if args.command == "console":
        from pickhero.audio.console import options_from_args, run_console_test

        try:
            run_console_test(options_from_args(args))
        except KeyboardInterrupt:
            print("\nStopped.")
        sys.exit(0)

    config = Config.load()
    _resolve_songs_dir(config)
    from pickhero.ui.app import App
    App(config=config).run()


if __name__ == "__main__":
    main()
