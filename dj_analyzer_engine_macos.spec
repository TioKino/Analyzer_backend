# -*- mode: python ; coding: utf-8 -*-
"""
DJ ANALYZER PRO - PyInstaller Build Spec (macOS, universal2)
=============================================================
Compila el backend FastAPI en un binario standalone para macOS, target
`universal2` (arm64 + x86_64 en un solo binario).

Uso:
  pyinstaller dj_analyzer_engine_macos.spec

Requisitos previos (en una Mac, no cross-compila):
  pip install pyinstaller
  pip install -r requirements.txt
  brew install ffmpeg                 # universal2 desde Homebrew
  # O compilar ffmpeg estaticamente para universal2 si quieres evitar
  # arrastrar las dylibs de Homebrew.

Output:
  dist/dj_analyzer_engine/dj_analyzer_engine
  dist/dj_analyzer_engine/_internal/...
  dist/dj_analyzer_engine/_internal/ffmpeg

El bundle completo (la carpeta `dj_analyzer_engine/`) se copia despues a
`DJ Analyzer.app/Contents/Resources/analyzer/dj_analyzer_engine/` durante
el build de Flutter (ver `Analyzer/macos/scripts/build_macos_local_engine.sh`).

NOTA: este spec es solo para distribucion DMG / Developer ID notarizada.
El build Mac App Store NO incluye el engine (sandbox + entitlements
incompatibles con PyInstaller + numpy/scipy).
"""

import os
import sys
import shutil
from PyInstaller.utils.hooks import collect_all

if sys.platform != 'darwin':
    raise SystemExit(
        "[SPEC] dj_analyzer_engine_macos.spec solo funciona en macOS. "
        "Para Windows usa dj_analyzer_engine.spec."
    )

block_cipher = None

# Recopilar todos los submodulos de las libs cientificas (tienen muchos
# imports dinamicos que PyInstaller no detecta solo).
librosa_datas, librosa_binaries, librosa_hiddenimports = collect_all('librosa')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')

# Modulos locales del proyecto (no son packages, son .py sueltos).
local_modules = [
    'main.py',
    'config.py',
    'database.py',
    'models.py',
    'validation.py',
    'artwork_and_cuepoints.py',
    'audd_helper.py',
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
    'community_cues_endpoint.py',
    'chromaprint_helper.py',
]

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

datas = []
for mod in local_modules:
    if os.path.exists(mod):
        datas.append((mod, '.'))
for mod in route_modules:
    if os.path.exists(mod):
        datas.append((mod, 'routes' if 'routes/' in mod else '.'))

if os.path.isdir('data'):
    for f in os.listdir('data'):
        fpath = os.path.join('data', f)
        if os.path.isfile(fpath):
            datas.append((fpath, 'data'))

datas += librosa_datas
datas += scipy_datas
datas += soundfile_datas
datas += numpy_datas

# Localizar ffmpeg. En Apple Silicon Homebrew lo instala en /opt/homebrew,
# en Intel en /usr/local. Si tienes una build estatica universal2 puesta
# manualmente en `./ffmpeg`, esa gana.
ffmpeg_path = None
for candidate in [
    './ffmpeg',
    '/opt/homebrew/bin/ffmpeg',
    '/usr/local/bin/ffmpeg',
]:
    if os.path.isfile(candidate):
        ffmpeg_path = candidate
        break
if ffmpeg_path is None:
    ffmpeg_path = shutil.which('ffmpeg')

binaries = []
if ffmpeg_path:
    binaries.append((ffmpeg_path, '.'))
    print(f"[SPEC] FFmpeg encontrado: {ffmpeg_path}")
    # Aviso sobre arquitecturas: si el ffmpeg de Homebrew NO es universal2
    # vas a tener que firmar dos builds (arm64 + x86_64) separadas o
    # compilar ffmpeg estaticamente. PyInstaller mete el binario tal cual.
    print("[SPEC] AVISO: si ffmpeg no es universal2, el .app universal2 "
          "fallara en la arquitectura no cubierta. Verifica con:")
    print(f"[SPEC]   lipo -info {ffmpeg_path}")
else:
    print("[SPEC] ADVERTENCIA: FFmpeg NO encontrado. Los previews no funcionaran.")
    print("[SPEC] Instala con `brew install ffmpeg` o pon un binario en ./ffmpeg")

binaries += librosa_binaries
binaries += scipy_binaries
binaries += soundfile_binaries
binaries += numpy_binaries

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
        'audd_helper',
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
        'community_cues_endpoint',
        'chromaprint_helper',

        # Routes
        'routes',
        'routes.admin',
        'routes.admin_panel',
        'routes.community',
        'routes.library',
        'routes.media',
        'routes.preview',
        'routes.search',
    ] + librosa_hiddenimports + scipy_hiddenimports + soundfile_hiddenimports + numpy_hiddenimports,
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
    # UPX no funciona bien en macOS con codesign, lo dejamos desactivado
    # para evitar que el bootloader pierda la firma al desempaquetar.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    # Por defecto = host arch (arm64 en Apple Silicon, x86_64 en Intel).
    # Para shippear a ambos hay que:
    #   ENGINE_TARGET_ARCH=universal2 pyinstaller dj_analyzer_engine_macos.spec
    # pero esto requiere wheels Python universal2 + ffmpeg universal2 (no
    # los de Homebrew, que son arm64-only en Apple Silicon). Si no las
    # tenes la build falla con "incompatible architecture" en lipo.
    # Plan razonable: shippear arm64 inicial, agregar x86_64 si hay demanda.
    target_arch=os.environ.get('ENGINE_TARGET_ARCH') or None,
    # codesign aplicado fuera de PyInstaller en sign_engine.sh para tener
    # control granular sobre orden de firmas y hashes de framework.
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='dj_analyzer_engine',
)
