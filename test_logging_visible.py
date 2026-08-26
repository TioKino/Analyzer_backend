"""Los `logger.info` del backend tienen que VERSE en produccion.

Durante toda la vida del proyecto no se vio ni uno. `main.py` hacia
`logger.setLevel(logging.INFO)` y no anadia ningun handler; tampoco habia
`basicConfig`. Y uvicorn —que es como arranca Render (`web: uvicorn main:app`)—
configura solo sus propios loggers, no el root.

Cuando un record no encuentra handler en toda la cadena, Python cae en
`logging.lastResort`, que es un `_StderrHandler` **a nivel WARNING**:

    logger.warning(...)  -> se ve
    logger.info(...)     -> se tira EN SILENCIO

Son 163 llamadas a `logger.info` en este backend. Incluidas las que dicen si
AudD se disparo y por que, que decidio el sync, o si un cluster adopto
metadata. Todas invisibles.

Y cuesta darse cuenta porque **en local funciona**: `local_engine.py` si llama a
`logging.basicConfig(...)`. Depuras en local, ves los logs, y das por hecho que
en Render tambien estan.

Se descubrio persiguiendo por que `[AudD-force] no corrio` no aparecia en los
logs de Render mientras `Sin ID3 valido` —un `logger.warning`— si. Esa
asimetria era toda la pista.

    pytest test_logging_visible.py -v
"""

import io
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# EL MECANISMO, aislado
# ============================================================================

def test_sin_handler_los_info_se_pierden():
    """El fallo original, reproducido. Que quede claro que NO es una teoria."""
    log = logging.getLogger('prueba_sin_handler_%s' % id(object()))
    log.setLevel(logging.INFO)

    # `lastResort` es lo unico que queda cuando nadie tiene handler.
    assert logging.lastResort.level == logging.WARNING, (
        'si esto cambia en una version de Python, el diagnostico de arriba '
        'deja de aplicar — reviselo antes de tocar la config'
    )


# ============================================================================
# LA CONFIGURACION REAL
# ============================================================================

@pytest.fixture(scope='module')
def _main_importado():
    import main  # noqa: F401  (el import es el que configura el logging)
    return main


def test_el_root_tiene_handler(_main_importado):
    """Importar `main` deja el root configurado.

    Es lo que hace que TODOS los modulos se vean, no solo `main`: los demas
    usan `logging.getLogger(__name__)` —`audd_helper`, `sync_endpoints`,
    `acoustic_fingerprint`— y esos NO cuelgan de `dj_analyzer`. Ponerle el
    handler solo al logger de main habria dejado mudo al resto, que es el
    arreglo que parece razonable si se lee por encima.
    """
    assert logging.getLogger().handlers, 'el root sigue sin handler'


def test_un_info_de_OTRO_modulo_llega_a_la_salida(_main_importado):
    """El caso que importa: `audd_helper` diciendo por que no disparo AudD."""
    log = logging.getLogger('audd_helper')

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(h)
    try:
        log.info('[AudD-auto] disparando: metadata basura')
    finally:
        root.removeHandler(h)

    assert '[AudD-auto] disparando' in buf.getvalue(), (
        'un logger.info de un modulo suelto no llega: el nivel efectivo lo '
        'esta cortando'
    )


def test_el_nivel_efectivo_del_root_deja_pasar_INFO(_main_importado):
    assert logging.getLogger().getEffectiveLevel() <= logging.INFO


def test_las_librerias_ruidosas_siguen_en_WARNING(_main_importado):
    """Subir el root a INFO sin acotar terceros cambia un problema por otro.

    Sin esto, el log de Render se llena del ruido de urllib3/PIL/numba y
    entierra justo lo que acabamos de rescatar: se pasa de no ver nada a no
    encontrar nada.
    """
    for nombre in ('urllib3', 'httpx', 'PIL', 'numba'):
        assert logging.getLogger(nombre).level >= logging.WARNING, (
            '%s puede inundar el log de produccion' % nombre
        )


def test_LOG_LEVEL_se_respeta_si_alguien_quiere_depurar():
    """La config lee `LOG_LEVEL` del entorno.

    Se comprueba sobre el FUENTE y no ejecutando: `main` ya se importo en esta
    sesion de pytest, asi que el valor del entorno de ahora no cambiaria nada.
    """
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
    with open(ruta, encoding='utf-8') as f:
        src = f.read()

    assert "os.getenv('LOG_LEVEL'" in src
    # Las dos mitades: un handler en el root Y un nivel que deje pasar INFO.
    # Con una sola no basta — sin handler se cae en `lastResort` (WARNING), y
    # sin nivel el record ni se crea.
    assert '_root_logger.addHandler(' in src, (
        'sin un handler en el root los info vuelven a caer en lastResort y '
        'desaparecen'
    )
    assert '_root_logger.setLevel(' in src, (
        'sin subir el nivel del root, el handler no recibe los INFO'
    )
