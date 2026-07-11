#!/bin/bash
# Build PickHero AppImage for Linux x86_64.
# Requires: pip install python-appimage
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

# Build the AppImage using python-appimage.
python-appimage build app -p 3.12 packaging/appimage/

OUTPUT="${ROOT_DIR}/PickHero-x86_64.AppImage"

if [[ ! -f "${OUTPUT}" ]]; then
    echo "ERROR: AppImage not found at ${OUTPUT}" >&2
    exit 1
fi

chmod +x "${OUTPUT}"
echo "Built: ${OUTPUT}"
