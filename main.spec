# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

local_hiddenimports = [
    'analysis',
    'app',
    'db_io',
    'models',
    'optimization',
    'plotting',
    'scipy',
    'scipy._lib',
    'scipy._lib.messagestream',
    'scipy.ndimage',
    'scipy.ndimage._filters',
    'sqlite3',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
] + collect_submodules('scipy.ndimage')


a = Analysis(
    ['needle_hook_data_plot.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('database', 'database'),
        ('README.md', '.'),
        ('requirements.txt', '.'),
    ],
    hiddenimports=local_hiddenimports,
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
    a.binaries,
    a.datas,
    [],
    exclude_binaries=False,
    name='needle_hook_data_plot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
