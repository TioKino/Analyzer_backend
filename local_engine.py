"""
DJ ANALYZER PRO - Motor Local de Análisis
==========================================
Este script arranca el backend FastAPI en localhost:8765
para que la app Flutter lo use como motor de análisis local.

Mismo código que el backend de Render, pero corriendo en tu PC.
Resultado: análisis 10x más rápido, funciona offline.

Uso:
  python local_engine.py          (desarrollo)
  dj_analyzer_engine.exe          (compilado con PyInstaller)
"""

import logging
import sys
import os
import signal
import shutil

# Monkey-patch global de subprocess.Popen — Windows only.
#
# Motivo: librosa.load() usa audioread internamente, que para mp3/m4a
# llama a `subprocess.Popen(['ffmpeg', ...])` SIN pasar creationflags.
# En Windows eso hace que cada decode (uno por track durante /analyze)
# abra una consola transitoria visible, produciendo un flash de
# PowerShell que el usuario ve por cada track al analizar.
#
# El cliente Flutter ya invoca SUS subprocesses con CREATE_NO_WINDOW via
# FFI Win32, pero no podemos controlar los subprocesses que el local
# engine de Python lanza internamente (audioread/librosa son
# dependencias). Parchear Popen al inicio del proceso garantiza que
# CUALQUIER subprocess que se cree heredara el flag, da igual desde que
# libreria salga.
#
# Tiene que ir ANTES del primer `import subprocess` real de cualquier
# modulo, asi que esta justo aqui arriba — antes incluso de configurar
# logging — para asegurar que se aplica a TODA la cadena de imports.
if sys.platform == 'win32':
    import subprocess as _sp
    _CREATE_NO_WINDOW = 0x08000000
    _orig_popen_init = _sp.Popen.__init__

    def _patched_popen_init(self, *args, **kwargs):
        # Combinamos en vez de sobreescribir por si el caller ya pasaba
        # flags propios (DETACHED_PROCESS, CREATE_NEW_PROCESS_GROUP, etc).
        kwargs['creationflags'] = (
            kwargs.get('creationflags', 0) | _CREATE_NO_WINDOW
        )
        return _orig_popen_init(self, *args, **kwargs)

    _sp.Popen.__init__ = _patched_popen_init

# Determinar directorio base antes de configurar logging
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    os.chdir(BASE_DIR)
    sys.path.insert(0, sys._MEIPASS)

    # PyInstaller en modo windowed (runw.exe) deja sys.stdout/sys.stderr como
    # None porque no hay consola. Cualquier print() legacy o cualquier libreria
    # que llame a sys.stderr.isatty() revienta con AttributeError. Redirigimos
    # a devnull (los logs utiles van al engine.log via FileHandler).
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

# DATA_DIR: directorio writable donde van log + DB + caches.
# - macOS frozen: BASE_DIR vive dentro del .app, read-only en /Applications.
#   Redirigimos a ~/Library/Application Support/DJ Analyzer/ (convencion
#   Apple para datos persistentes de apps).
# - Windows frozen: BASE_DIR es la carpeta de instalacion. Windows aplica
#   VirtualStore redirect transparente a %LOCALAPPDATA%\VirtualStore\... si
#   Program Files es read-only, asi que dejarlo en BASE_DIR sigue
#   funcionando como hasta ahora.
# - Dev (no frozen): BASE_DIR es Analyzer_backend/, writable, no tocamos.
if getattr(sys, 'frozen', False) and sys.platform == 'darwin':
    DATA_DIR = os.path.expanduser('~/Library/Application Support/DJ Analyzer')
    os.makedirs(DATA_DIR, exist_ok=True)
else:
    DATA_DIR = BASE_DIR

# Configurar logging: archivo + consola (si hay)
log_file = os.path.join(DATA_DIR, 'engine.log')
handlers = [logging.FileHandler(log_file, encoding='utf-8')]
if not getattr(sys, 'frozen', False):
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=handlers,
)
logger = logging.getLogger(__name__)

# Resolver FFmpeg a ruta absoluta antes de importar main.
# main.py y preview_generator.py usan `os.environ.get('FFMPEG_BIN', 'ffmpeg')`
# en todas sus subprocess.run. La ruta absoluta evita WinError 448 en
# Windows 11 24H2+ cuando el PATH contiene reparse points (OneDrive,
# junctions, symlinks). El .spec empaqueta ffmpeg.exe junto al engine.
#
# PyInstaller 6+ pone los binaries en `_internal/` dentro del bundle
# one-folder, no en la raiz. Por eso la primera candidatura que miramos
# tras el cwd es `BASE_DIR/_internal/ffmpeg.exe`.
_ffmpeg_candidates = [
    os.path.join(BASE_DIR, '_internal', 'ffmpeg.exe'),   # PyInstaller 6+ Windows
    os.path.join(BASE_DIR, '_internal', 'ffmpeg'),        # PyInstaller 6+ Linux/Mac
    os.path.join(BASE_DIR, 'ffmpeg.exe'),                 # PyInstaller <6 o manual
    os.path.join(BASE_DIR, 'ffmpeg'),
    shutil.which('ffmpeg.exe'),                           # PATH Windows
    shutil.which('ffmpeg'),                               # PATH Linux/Mac
]
for _c in _ffmpeg_candidates:
    if _c and os.path.isfile(_c):
        os.environ['FFMPEG_BIN'] = _c
        logger.info(f"FFmpeg resuelto a: {_c}")
        break
else:
    logger.warning("ffmpeg no encontrado — previews y waveforms fallaran")

# Configurar variables de entorno ANTES de importar main
os.environ['PORT'] = '8000'
os.environ['HOST'] = '0.0.0.0'
os.environ['DEBUG'] = 'false'
os.environ['LOCAL_ENGINE'] = 'true'  # Señal para main.py: modo local, CPU del usuario
os.environ['DATABASE_PATH'] = os.path.join(DATA_DIR, 'local_analysis.db')
os.environ['ARTWORK_CACHE_DIR'] = os.path.join(DATA_DIR, 'artwork_cache')
os.environ['PREVIEWS_DIR'] = os.path.join(DATA_DIR, 'previews_cache')
# SYNC_DB_PATH: routes/admin_panel.py por defecto apunta a /data/sync.db
# (path de Render/Linux). En el motor local Mac/Windows ese path no existe
# y los endpoints admin/* tiran 500 con `unable to open database file`.
# Lo redirigimos a un archivo dentro del DATA_DIR del motor para que el
# panel admin funcione contra el motor local. Si el archivo no existe
# todavia se crea vacio en la primera conexion.
os.environ['SYNC_DB_PATH'] = os.path.join(DATA_DIR, 'local_sync.db')

# RENDER_SYNC_URL: el motor local lo usa para hacer push de previews/artwork
# al Render. Flutter (LocalEngineService) lo pasa cuando lanza el engine,
# pero si el engine arranca standalone (usuario hace doble click en el .exe,
# o el proceso queda corriendo de una sesión anterior), no llegaría. Default
# al Render de producción para que push funcione siempre.
if not os.environ.get('RENDER_SYNC_URL'):
    os.environ['RENDER_SYNC_URL'] = 'https://dj-analyzer-api.onrender.com'

# Crear directorios si no existen
os.makedirs(os.environ['ARTWORK_CACHE_DIR'], exist_ok=True)
os.makedirs(os.environ['PREVIEWS_DIR'], exist_ok=True)


def main():
    logger.info("=" * 60)
    logger.info("  DJ ANALYZER PRO - Motor Local")
    logger.info(f"  Puerto: {os.environ['PORT']}")
    logger.info(f"  Base: {BASE_DIR}")
    logger.info(f"  Data: {DATA_DIR}")
    logger.info(f"  DB: {os.environ['DATABASE_PATH']}")
    logger.info("=" * 60)

    import uvicorn

    # Importar la app FastAPI desde main.py
    from main import app

    # Pre-init sync.db (crea sync_items, users, etc) para que el panel
    # admin pueda consultar incluso si no hay /sync/* requests previas
    # — caso tipico del motor local en Mac/Windows fresh.
    try:
        from sync_endpoints import _get_conn as _init_sync_db
        _init_sync_db()
        logger.info("  Sync DB inicializada: %s", os.environ['SYNC_DB_PATH'])
    except Exception as e:
        logger.warning("Sync DB pre-init fallo (admin panel podria 500): %s", e)

    # Manejar Ctrl+C limpiamente
    def handle_exit(sig, frame):
        logger.info("Motor Local apagando...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Arrancar servidor
    # log_config=None: le decimos a uvicorn que NO configure su propio logging.
    # En PyInstaller windowed su default llama a sys.stderr.isatty() al
    # construir el formatter y revienta porque stderr es None. Con None,
    # uvicorn usa el logging ya configurado arriba (FileHandler -> engine.log).
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        log_config=None,
    )


if __name__ == "__main__":
    main()
