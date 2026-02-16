"""PickHero application entry point."""

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


def main():
    if "--console" in sys.argv:
        # Phase 1 console demo for audio testing
        from pickhero.audio.input import run_console_demo
        try:
            run_console_demo()
        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)
    else:
        config = Config.load()
        _resolve_songs_dir(config)
        from pickhero.ui.app import App
        App(config=config).run()


if __name__ == "__main__":
    main()
