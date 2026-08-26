"""`/health` dice QUE commit esta corriendo.

Antes daba `version` y `uptime_seconds` y nada mas, asi que para saber si un
deploy concreto habia entrado habia que DEDUCIRLO por el uptime: «lleva 40
segundos levantado, debe de ser el mio». Falla en cuanto Render reinicia el
worker por su cuenta — y entonces no solo no te enteras, sino que te crees que
si.

Estaba anotado como abierto en PENDING desde el 2026-08-20.

De paso, `version` estaba escrita a mano en TRES sitios (el constructor de
FastAPI, `/` y `/health`), asi que subir una y olvidar otra dejaba dos numeros
distintos conviviendo sin que nada avisara. Es la misma forma que tenia
`X-Client-Version` en el cliente, clavado en '2.3.0' durante seis versiones.

    pytest test_health_dice_el_commit.py -v
"""

import importlib
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_AQUI = os.path.dirname(os.path.abspath(__file__))


def _fuente(nombre):
    with open(os.path.join(_AQUI, nombre), encoding='utf-8') as f:
        return f.read()


# ============================================================================
# UNA SOLA VERSION
# ============================================================================

def test_la_version_vive_en_UN_sitio():
    src = _fuente('main.py')
    # Una sola asignacion de la constante...
    assert len(re.findall(r'^API_VERSION = "', src, re.M)) == 1
    # ...y ni un literal suelto en los tres sitios que la usaban.
    assert 'version="2.9.9"' not in src
    assert '"version": "2.9.9"' not in src
    # Los tres consumidores la leen de la constante.
    assert 'version=API_VERSION' in src
    assert src.count('"version": API_VERSION') == 2


def test_los_tres_sitios_dicen_LO_MISMO():
    """El fallo que esto evita no es teorico: con tres literales, subir dos y
    olvidar uno deja `/` y `/health` contestando versiones distintas, y nadie
    se entera hasta que alguien compara."""
    import main
    importlib.reload(main)
    assert main.app.version == main.API_VERSION


# ============================================================================
# EL COMMIT
# ============================================================================

def test_el_commit_sale_del_entorno_de_Render(monkeypatch):
    monkeypatch.setenv('RENDER_GIT_COMMIT', 'abcdef1234567890abcdef')
    monkeypatch.setenv('RENDER_GIT_BRANCH', 'main')
    import main
    importlib.reload(main)

    # Cortado a 12: un sha entero no aporta nada y ensucia el volcado.
    assert main.DEPLOY_COMMIT == 'abcdef123456'
    assert main.DEPLOY_BRANCH == 'main'


def test_fuera_de_Render_es_None_y_NO_una_cadena_inventada(monkeypatch):
    """None significa «no lo se». Un `'unknown'` o un `''` convertirian un
    hueco en un dato — el mismo fallo que la ficha de Info del cliente tenia
    con `bpm_source`, que caia a `'analysis'` y afirmaba que lo habia medido el
    backend."""
    monkeypatch.delenv('RENDER_GIT_COMMIT', raising=False)
    monkeypatch.delenv('RENDER_GIT_BRANCH', raising=False)
    import main
    importlib.reload(main)

    assert main.DEPLOY_COMMIT is None
    assert main.DEPLOY_BRANCH is None


def test_una_variable_VACIA_tambien_es_None(monkeypatch):
    """Render puede exportarla vacia. `''[:12]` es `''`, que es falsy pero NO
    es None: sin el `or None` final, `/health` contestaria `"commit": ""` y eso
    se lee como un commit que existe y esta en blanco."""
    monkeypatch.setenv('RENDER_GIT_COMMIT', '')
    import main
    importlib.reload(main)
    assert main.DEPLOY_COMMIT is None


# ============================================================================
# LA RESPUESTA
# ============================================================================

@pytest.mark.asyncio
async def test_health_devuelve_commit_rama_y_arranque():
    import main
    importlib.reload(main)
    r = await main.health()

    for clave in ('status', 'version', 'commit', 'branch',
                  'uptime_seconds', 'started_at', 'checks'):
        assert clave in r, f'falta {clave} en /health'

    # `started_at` en absoluto y con zona: `uptime_seconds` solo dice CUANTO
    # lleva, y para cruzarlo con la hora de un merge hace falta el instante.
    assert r['started_at'].endswith('+00:00'), 'started_at sin zona horaria'


@pytest.mark.asyncio
async def test_health_no_delata_rutas_de_disco():
    """/health es PUBLICO. El detalle de un fallo de SQLite lleva la ruta del
    fichero ('unable to open database file: /data/...') y eso va al log, no al
    cliente. Se comprueba que la respuesta solo trae el estado."""
    import main
    importlib.reload(main)
    r = await main.health()
    assert r['checks']['database'] in ('ok', 'error')
    assert '/data' not in str(r['checks'])
