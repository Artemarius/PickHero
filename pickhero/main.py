"""PickHero application entry point."""

import sys


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
        from pickhero.ui.app import App
        App().run()
