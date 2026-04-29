# -*- mode: python ; coding: utf-8 -*-

local_hiddenimports = [
    'analysis',
    'app',
    'db_io',
    'models',
    'optimization',
    'plotting',
    'sqlite3',
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
]


a = Analysis(
    ['needle_hook_data_plot.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('database', 'database'),
        ('README.md', '.'),
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
