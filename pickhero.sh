#!/usr/bin/env bash
# Launcher for PickHero on Linux
cd "$(dirname "$0")" && exec venv/bin/python -m pickhero "$@"
