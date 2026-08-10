"""
Configuración de pytest para DJ ANALYZER
"""

import pytest
import sys
import os
import tempfile

# Añadir el directorio del proyecto al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar variables de entorno para tests
os.environ.setdefault('AUDD_API_TOKEN', 'test_token')
os.environ.setdefault('DISCOGS_TOKEN', 'test_token')
os.environ.setdefault('BASE_URL', 'http://localhost:8000')

# ── Aislar el disco: NINGÚN test debe tocar /data ────────────────────────────
#
# `sync_endpoints._DB_PATH` se evalúa en el IMPORT del módulo y cae por defecto
# a `/data/sync.db` (la ruta del disco persistente de Render). Cada fichero de
# test hacía su propio `os.environ.setdefault("SYNC_DB_PATH", ...)` dentro de
# una fixture, pero eso llega TARDE: pytest importa los módulos de test durante
# la COLECCIÓN, antes de ejecutar ninguna fixture, así que el primer módulo que
# hiciera `import main` a nivel de fichero fijaba `_DB_PATH` a `/data/sync.db`
# para toda la sesión.
#
# En una máquina donde `/data` existe y es escribible (contenedor de dev como
# root) esto PASABA — escribiendo en la BD real, que además es la de Render si
# alguien corre la suite en el servidor. En el runner de Actions, que no es
# root y no tiene `/data`, reventaba con
# `PermissionError: [Errno 13] Permission denied: '/data'` (5 fallos en
# test_sync_pagination.py). Green en local, rojo en CI, por el entorno.
#
# conftest.py se importa ANTES de colectar los tests, así que aquí sí llega a
# tiempo. Se fija con setdefault para que quien quiera apuntar a una BD
# concreta pueda hacerlo desde fuera (`SYNC_DB_PATH=... pytest`).
_TEST_TMP = tempfile.mkdtemp(prefix="djanalyzer-tests-")
os.environ.setdefault('SYNC_DB_PATH', os.path.join(_TEST_TMP, 'sync.db'))
os.environ.setdefault('DATABASE_PATH', os.path.join(_TEST_TMP, 'analysis.db'))
os.environ.setdefault('PREVIEWS_DIR', os.path.join(_TEST_TMP, 'previews'))
os.environ.setdefault('ARTWORK_CACHE_DIR', os.path.join(_TEST_TMP, 'artwork'))
os.environ.setdefault('FPCALC_CACHE_DIR', os.path.join(_TEST_TMP, 'bin'))
# La suite no debe comportarse como producción (docs cerradas, auth exigida…).
os.environ.pop('RENDER', None)


@pytest.fixture(scope="session")
def test_config():
    """Configuración global para tests"""
    return {
        'base_url': 'http://localhost:8000',
        'test_mode': True
    }


def pytest_configure(config):
    """Configuración inicial de pytest"""
    config.addinivalue_line(
        "markers", "slow: marca tests que tardan mucho"
    )
    config.addinivalue_line(
        "markers", "integration: tests de integración"
    )
    config.addinivalue_line(
        "markers", "unit: tests unitarios"
    )
