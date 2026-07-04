#!/bin/bash
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-wayland}
exec {{ python-executable }} -I ${APPDIR}/opt/python{{ python-version }}/bin/pickhero "$@"
