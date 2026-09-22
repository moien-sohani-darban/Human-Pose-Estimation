# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ENGINE_ROOT = Path(SPECPATH).resolve()

a = Analysis(
    [str(ENGINE_ROOT / 'packaging' / 'sidecar_entry.py')],
    pathex=[str(ENGINE_ROOT)],
    binaries=[],
    datas=[],
    # TorchVision 0.29 renamed its native extension modules; the current
    # PyInstaller hook still requests the legacy _C/image names.
    hiddenimports=["torchvision._C_stable", "torchvision.image_stable"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hpe-python-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='hpe-python-sidecar',
)
