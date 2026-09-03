"""De los tracks que SIGUEN entrando sin huella, ¿cual es un bug?

`by_engine_last_30d` separo el legado de lo reciente y esa fue la mitad grande
del problema. Pero deja los tres candidatos a la via abierta en dos cubos, y
dos de ellos caen JUNTOS pidiendo acciones opuestas:

  El fallback de `/analyze` —el que se crea cuando librosa no puede con el
  fichero— escribe la fila con bpm=0, sin key y SIN llamar a `_attach_acoustic`.
  Sale sin huella **a proposito**: un analisis fallido no debe sembrar clusters
  con una duracion basura. No hay nada que arreglar.

  Un analisis que SI salio bien y aun asi no trajo chromaprint es otra cosa.
  `_attach_acoustic` corre incondicionalmente sobre el fichero subido, asi que
  si falta la huella es que `fpcalc` fallo sobre ese audio y el best-effort se
  trago la excepcion. **Ahi si hay bug.**

Con un solo contador, ampliar el backfill parece la respuesta en los dos casos
—y en el primero seria achicar agua sin tapar la via, y en el segundo ni eso.

Y hay una trampa de lectura al lado: el fallback **no sella `engine_source`**
(decision consciente, esta escrita en `/analyze`), asi que cae en `unknown`. Un
`unknown` RECIENTE no significa «no sabemos de donde vino».

    pytest test_gap_reciente_bug_o_por_diseno.py -v
"""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from database import AnalysisDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield AnalysisDB(db_path=path)
    try:
        os.unlink(path)
    except OSError:
        pass


def _hace(dias):
    return (datetime.now() - timedelta(days=dias)).isoformat()


def _track(db, tid, *, bpm=128.0, key='Am', chromaprint=None,
           engine='render', dias=1):
    """Una fila de `tracks` con lo que mira el reparto."""
    db.save_track({
        'id': tid,
        'filename': f'{tid}.mp3',
        'duration': 300.0,
        'bpm': bpm,
        'key': key,
        'camelot': '8A' if key else None,
        'energy_dj': 5,
        'genre': 'Techno',
        'track_type': 'club',
        'chromaprint': chromaprint,
        'engine_source': engine,
        'analyzed_at': _hace(dias),
        'fingerprint': tid,
    })


def _fallback_de_analyze(db, tid, *, dias=1):
    """Lo que escribe el fallback de `/analyze` cuando librosa no puede.

    bpm=0, sin key, y **sin `engine_source`**: no lo sella a proposito, porque
    no representa un analisis real. Ver el comentario en `/analyze`.
    """
    _track(db, tid, bpm=0, key=None, engine=None, dias=dias)


# ============================================================================
# LO QUE SEPARA
# ============================================================================

def test_el_fallback_fallido_NO_cuenta_como_bug(db):
    _fallback_de_analyze(db, 'roto1')
    _fallback_de_analyze(db, 'roto2')
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d'] == {
        'failed_fallback': 2,
        'analyzed_ok': 0,
    }


def test_un_analisis_BUENO_sin_huella_SI_es_un_bug(db):
    """`_attach_acoustic` corre incondicionalmente sobre el fichero subido. Si
    el analisis salio y la huella no esta, algo paso."""
    _track(db, 'ok1', chromaprint=None)
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d'] == {
        'failed_fallback': 0,
        'analyzed_ok': 1,
    }


def test_mezclados_no_se_tapan_el_uno_al_otro(db):
    """El caso real: los dos a la vez. Con un solo numero, el que decide queda
    enterrado bajo el otro — que es exactamente lo que hacia `by_engine` con el
    legado."""
    for i in range(9):
        _fallback_de_analyze(db, f'roto{i}')
    _track(db, 'ok1', chromaprint=None)
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d']['failed_fallback'] == 9
    assert r['by_outcome_last_30d']['analyzed_ok'] == 1


def test_los_que_SI_tienen_huella_no_entran(db):
    """El reparto es de los que estan SIN huella. Un track con chromaprint no
    es ni un bug ni un fallback: no es nada, no sale."""
    _track(db, 'con', chromaprint='AQADtEmk...')
    _fallback_de_analyze(db, 'roto1')
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d'] == {
        'failed_fallback': 1,
        'analyzed_ok': 0,
    }


def test_el_legado_no_entra_en_el_reparto_reciente(db):
    """La razon de ser de todo esto: el legado es dos ordenes de magnitud
    mayor. Si se colara aqui, taparia el numero igual que tapaba el de
    `by_engine`."""
    _track(db, 'viejo', chromaprint=None, dias=200)
    _track(db, 'nuevo', chromaprint=None, dias=2)
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d']['analyzed_ok'] == 1
    # ...pero el total sigue contandolos a los dos: son dos cosas distintas.
    assert r['without_chromaprint'] == 2


def test_bpm_cero_pero_CON_key_no_es_el_fallback(db):
    """El predicado son las DOS columnas. Una fila con bpm=0 y key puesta no
    la escribio ese fallback —que no pone key jamas— y llamarla «fallo por
    diseño» seria esconder un caso que no se ha mirado."""
    _track(db, 'raro', bpm=0, key='Am', chromaprint=None)
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d']['failed_fallback'] == 0
    assert r['by_outcome_last_30d']['analyzed_ok'] == 1


# ============================================================================
# LA TRAMPA DE LECTURA DE AL LADO
# ============================================================================

def test_el_fallback_cae_en_unknown_y_por_eso_hace_falta_el_otro_eje(db):
    """No es un bug del sellado: `/analyze` NO marca `engine_source` en el
    fallback a proposito, porque esa fila no representa un analisis real.

    Pero la consecuencia al leer el panel es que un `unknown` RECIENTE no
    quiere decir «origen sin sellar, legado» — quiere decir esto. Los dos ejes
    juntos lo dicen; `by_engine_last_30d` solo, no.
    """
    _fallback_de_analyze(db, 'roto1')
    r = db.acoustic_gap_breakdown()
    assert r['by_engine_last_30d'] == {'unknown': 1}
    assert r['by_outcome_last_30d']['failed_fallback'] == 1


def test_los_dos_ejes_cuentan_la_MISMA_poblacion(db):
    """Si no suman lo mismo, uno de los dos esta filtrando de mas y las dos
    lecturas se contradicen sin que nadie sepa cual creer."""
    _fallback_de_analyze(db, 'roto1')
    _track(db, 'ok1', chromaprint=None)
    _track(db, 'ok2', chromaprint=None, engine='local_engine')
    _track(db, 'viejo', chromaprint=None, dias=200)  # fuera de los 30 dias
    r = db.acoustic_gap_breakdown()
    assert sum(r['by_engine_last_30d'].values()) == 3
    assert sum(r['by_outcome_last_30d'].values()) == 3


def test_una_bd_vacia_no_inventa_nada(db):
    r = db.acoustic_gap_breakdown()
    assert r['by_outcome_last_30d'] == {
        'failed_fallback': 0,
        'analyzed_ok': 0,
    }


# ============================================================================
# EL CABLEADO
# ============================================================================

def test_el_script_del_embudo_lo_imprime():
    """El dato existiendo y no enseñandose es el patron que mas veces ha
    mordido: `by_engine_last_30d` llevo una semana calculado y sin imprimir.

    `embudo.sh` vive en el repo del cliente, al lado. Si no esta —CI del
    backend solo—, no se inventa un verde.
    """
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'Analyzer', 'scripts', 'embudo.sh')
    if not os.path.exists(ruta):
        pytest.skip('el repo del cliente no esta al lado')
    with open(ruta, encoding='utf-8') as f:
        sh = f.read()
    assert 'by_outcome_last_30d' in sh
