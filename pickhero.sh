#!/usr/bin/env bash
# Launcher for PickHero on Linux
cd "$(dirname "$0")" && exec env PICKHERO_DEBUG_MATCH=1 venv/bin/python -m pickhero "$@"
