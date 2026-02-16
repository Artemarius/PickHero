# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for PickHero.

Build with:  pyinstaller pickhero.spec --noconfirm
Output:      dist/PickHero.exe  (single-file, windowed)
"""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

# ── Data files ──────────────────────────────────────────────────────────────
# sounddevice bundles the PortAudio shared library as package data.
# certifi provides the CA certificate bundle needed for HTTPS downloads
# (Songsterr search/download via urllib).
datas = []
datas += collect_data_files("sounddevice")
datas += collect_data_files("certifi")

# ── Native binaries / C extensions ──────────────────────────────────────────
# aubio  — C pitch/onset detection library
# pygame — SDL2 + mixer + image + font shared libraries
# numpy  — C core (multiarray, umath, etc.) — sometimes missed by analysis
binaries = []
binaries += collect_dynamic_libs("aubio")
binaries += collect_dynamic_libs("pygame")
binaries += collect_dynamic_libs("numpy")

# ── Analysis ────────────────────────────────────────────────────────────────
a = Analysis(
    ["pickhero/main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # Audio capture & detection
        "aubio",
        "numpy",
        "sounddevice",
        # UI
        "pygame",
        "pygame.midi",
        # Tab loading — the pip package is "pyguitarpro", but the import is "guitarpro"
        "guitarpro",
        # SSL certs for urllib HTTPS requests (Songsterr downloader)
        "certifi",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="PickHero",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
