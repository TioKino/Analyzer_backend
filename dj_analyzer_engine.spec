# -*- mode: python ; coding: utf-8 -*-
"""
DJ ANALYZER PRO - PyInstaller Build Spec
=========================================
Compila el backend FastAPI en un ejecutable standalone.

Uso:
  pyinstaller dj_analyzer_engine.spec

Requisitos previos:
  pip install pyinstaller
  pip install -r requirements.txt
  ffmpeg.exe en PATH o en esta carpeta
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Recopilar todos los submodulos de librosa (tiene muchos imports dinamicos)
librosa_datas, librosa_binaries, librosa_hiddenimports = collect_all('librosa')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')

# Modulos locales del proyecto (no son packages, son .py sueltos)
local_modules = [
    'main.py',
    'config.py',
    'database.py',
    'models.py',
    'validation.py',
    'artwork_and_cuepoints.py',
    'audio_helpers.py',
    'bpm_utils.py',
    'beatport.py',
    'genre_detection.py',
    'spectral_genre_classifier.py',
    'chunked_analyzer.py',
    'precision_analyzer.py',
    'preview_generator.py',
    'similar_tracks_endpoint.py',
    'search_analyzed_endpoint.py',
    'sync_endpoints.py',
    'api_config.py',
    'essentia_analyzer.py',
]

# Archivos de rutas (routes/)
route_modules = [
    'routes/__init__.py',
    'routes/admin.py',
    'routes/admin_panel.py',
    'routes/community.py',
    'routes/library.py',
    'routes/media.py',
    'routes/preview.py',
    'routes/search.py',
]

# Construir lista de datas (archivos .py locales que PyInstaller no detecta)
datas = []
for mod in local_modules:
    if os.path.exists(mod):
        datas.append((mod, '.'))
for mod in route_modules:
    if os.path.exists(mod):
        datas.append((mod, 'routes' if 'routes/' in mod else '.'))

# Agregar archivos de datos del proyecto
if os.path.isdir('data'):
    for f in os.listdir('data'):
        fpath = os.path.join('data', f)
        if os.path.isfile(fpath):
            datas.append((fpath, 'data'))

# Agregar datas de librosa, scipy, soundfile
datas += librosa_datas
datas += scipy_datas
datas += soundfile_datas

# Buscar ffmpeg.exe en el sistema
ffmpeg_path = None
for candidate in ['ffmpeg.exe', 'ffmpeg']:
    # Buscar en la carpeta actual
    if os.path.exists(candidate):
        ffmpeg_path = candidate
        break
    # Buscar en PATH
    import shutil
    found = shutil.which(candidate)
    if found:
        ffmpeg_path = found
        break

binaries = []
if ffmpeg_path:
    binaries.append((ffmpeg_path, '.'))
    print(f"[SPEC] FFmpeg encontrado: {ffmpeg_path}")
else:
    print("[SPEC] ADVERTENCIA: FFmpeg NO encontrado. Los previews no funcionaran.")
    print("[SPEC] Descarga ffmpeg.exe y ponlo en esta carpeta o en PATH.")

binaries += librosa_binaries
binaries += scipy_binaries
binaries += soundfile_binaries

a = Analysis(
    ['local_engine.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # FastAPI y servidor
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'fastapi.middleware',
        'fastapi.middleware.cors',
        'starlette',
        'starlette.responses',
        'starlette.routing',
        'starlette.middleware',
        'pydantic',
        'multipart',
        'python_multipart',

        # Audio
        'librosa',
        'librosa.core',
        'librosa.feature',
        'librosa.onset',
        'librosa.beat',
        'librosa.effects',
        'librosa.util',
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.fft',
        'soundfile',
        'mutagen',
        'mutagen.mp3',
        'mutagen.flac',
        'mutagen.mp4',
        'mutagen.id3',

        # HTTP y APIs
        'requests',
        'discogs_client',

        # Sistema
        'sqlite3',
        'json',
        'hashlib',
        'hmac',
        'tempfile',
        'shutil',
        'threading',
        'uuid',
        'base64',
        'math',
        'warnings',
        'gc',
        'logging',
        're',
        'signal',
        'datetime',
        'time',
        'pathlib',
        'random',
        'string',

        # Modulos locales
        'main',
        'config',
        'database',
        'models',
        'validation',
        'artwork_and_cuepoints',
        'audio_helpers',
        'bpm_utils',
        'beatport',
        'genre_detection',
        'spectral_genre_classifier',
        'chunked_analyzer',
        'precision_analyzer',
        'preview_generator',
        'similar_tracks_endpoint',
        'search_analyzed_endpoint',
        'sync_endpoints',
        'api_config',

        # Routes
        'routes',
        'routes.admin',
        'routes.admin_panel',
        'routes.community',
        'routes.library',
        'routes.media',
        'routes.preview',
        'routes.search',
    ] + librosa_hiddenimports + scipy_hiddenimports + soundfile_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'pytest_asyncio',
        'pytest_cov',
        'httpx',
        'essentia',
        'tkinter',
        'matplotlib',
        'IPython',
        'notebook',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dj_analyzer_engine',
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
    icon='../Analyzer/assets/icon/app_icon.ico' if os.path.exists('../Analyzer/assets/icon/app_icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='dj_analyzer_engine',
)
