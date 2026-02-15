"""PickHero application entry point."""

import sys


def main():
    print("PickHero — Guitar Practice Tool")
    print("Phase 1: Audio Detection")
    print()

    # Phase 1: run the console demo for pitch detection testing
    from pickhero.audio.input import run_console_demo
    try:
        run_console_demo()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
