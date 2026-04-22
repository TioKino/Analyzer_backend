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

# Configurar logging: archivo + consola (si hay)
log_file = os.path.join(BASE_DIR, 'engine.log')
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

# Configurar variables de entorno ANTES de importar main
os.environ['PORT'] = '8000'
os.environ['HOST'] = '0.0.0.0'
os.environ['DEBUG'] = 'false'
os.environ['LOCAL_ENGINE'] = 'true'  # Señal para main.py: modo local, CPU del usuario
os.environ['DATABASE_PATH'] = os.path.join(BASE_DIR, 'local_analysis.db')
os.environ['ARTWORK_CACHE_DIR'] = os.path.join(BASE_DIR, 'artwork_cache')
os.environ['PREVIEWS_DIR'] = os.path.join(BASE_DIR, 'previews_cache')

# Crear directorios si no existen
os.makedirs(os.environ['ARTWORK_CACHE_DIR'], exist_ok=True)
os.makedirs(os.environ['PREVIEWS_DIR'], exist_ok=True)


def main():
    logger.info("=" * 60)
    logger.info("  DJ ANALYZER PRO - Motor Local")
    logger.info(f"  Puerto: {os.environ['PORT']}")
    logger.info(f"  Base: {BASE_DIR}")
    logger.info(f"  DB: {os.environ['DATABASE_PATH']}")
    logger.info("=" * 60)

    import uvicorn

    # Importar la app FastAPI desde main.py
    from main import app

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
