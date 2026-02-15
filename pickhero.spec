# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for PickHero."""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
)

block_cipher = None

# Collect runtime data/libraries that PyInstaller misses
datas = []
datas += collect_data_files("sounddevice")  # includes PortAudio DLL
datas += collect_data_files("certifi")       # SSL certs for Songsterr downloads

binaries = []
binaries += collect_dynamic_libs("aubio")
binaries += collect_dynamic_libs("pygame")

a = Analysis(
    ["pickhero/main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "aubio",
        "numpy",
        "sounddevice",
        "pygame",
        "pygame.midi",
        "pyguitarpro",
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
