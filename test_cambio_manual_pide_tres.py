"""Un cambio hecho A MANO llega a los demás DJs solo si lo comparten TRES
cuentas.

Decisión del owner, 2026-09-26: «un DJ puede estar equivocado, pero tres no».
Lo que se IMPORTA de Rekordbox/Traktor/VirtualDJ va por otro camino y basta
con uno (test_lo_importado_llega_a_todos.py).

Hasta ese día había dos puertas por debajo de tres:

  - La rejilla corregida a mano: el cliente aplicaba cualquier corrección con
    `contributors > 0`, o sea la de UN solo DJ. Y `validated` pedía dos
    aparatos —no cuentas— sin mirar si decían lo mismo: promediaba +0,5 y
    +0,9 BPM y salía +0,7, que no era de nadie.
  - El género: con dos votos se aplicaba como `suggestion_2`, que gana al
    análisis y a Discogs.

    pytest test_cambio_manual_pide_tres.py -v
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import AnalysisDB, MIN_CUENTAS_CAMBIO_MANUAL  # noqa: E402

FP = 'b' * 32
AQUI = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def db(tmp_path):
    d = AnalysisDB(db_path=str(tmp_path / 'analysis.db'))
    d.cuentas = {}
    d.cuentas_de = lambda ids: {i: d.cuentas[i] for i in ids if i in d.cuentas}
    return d


def _rejilla(db, device, ajuste=0.1, fase=0.01, original=128.0):
    db.submit_beat_grid_correction(FP, device, ajuste, fase, original)


def test_el_umbral_es_tres():
    assert MIN_CUENTAS_CAMBIO_MANUAL == 3


def test_uno_o_dos_djs_no_cambian_la_rejilla_de_nadie(db):
    _rejilla(db, 'a')
    r = db.get_community_beat_grid(FP)
    assert r['validated'] is False
    # A CERO, no solo `validated: False`: los clientes ya publicados aplican
    # cualquier ajuste distinto de cero que les llegue.
    assert r['bpm_adjust'] == 0.0 and r['beat_offset'] == 0.0

    _rejilla(db, 'b')
    r = db.get_community_beat_grid(FP)
    assert r['validated'] is False
    assert r['bpm_adjust'] == 0.0 and r['beat_offset'] == 0.0


def test_tres_djs_que_coinciden_si(db):
    _rejilla(db, 'a', ajuste=0.10, fase=0.010)
    _rejilla(db, 'b', ajuste=0.11, fase=0.012)
    _rejilla(db, 'c', ajuste=0.10, fase=0.011)
    r = db.get_community_beat_grid(FP)
    assert r['validated'] is True
    assert r['contributors'] == 3
    assert r['bpm_adjust'] == pytest.approx(0.1033, abs=1e-4)
    assert r['beat_offset'] == pytest.approx(0.011, abs=1e-6)


def test_tres_djs_que_dicen_cosas_distintas_no(db):
    _rejilla(db, 'a', ajuste=0.5)
    _rejilla(db, 'b', ajuste=0.9)
    _rejilla(db, 'c', ajuste=-0.3)
    r = db.get_community_beat_grid(FP)
    assert r['validated'] is False
    assert r['bpm_adjust'] == 0.0


def test_el_que_se_sale_no_estropea_la_media_de_los_que_coinciden(db):
    for d in 'abc':
        _rejilla(db, d, ajuste=0.10, fase=0.010)
    _rejilla(db, 'z', ajuste=0.90, fase=0.200)
    r = db.get_community_beat_grid(FP)
    assert r['validated'] is True
    assert r['contributors'] == 3
    assert r['bpm_adjust'] == pytest.approx(0.10)


def test_tres_aparatos_de_la_misma_cuenta_son_un_dj(db):
    db.cuentas = {'mac': 'u1', 'movil': 'u1', 'tablet': 'u1'}
    for d in ('mac', 'movil', 'tablet'):
        _rejilla(db, d)
    assert db.get_community_beat_grid(FP)['validated'] is False


def test_se_compara_el_bpm_final_no_el_ajuste(db):
    # Tres DJs llegan a 128,00 desde análisis distintos: el ajuste de cada uno
    # es distinto y el resultado es el mismo. Eso es coincidir.
    _rejilla(db, 'a', ajuste=0.10, original=127.90)
    _rejilla(db, 'b', ajuste=0.00, original=128.00)
    _rejilla(db, 'c', ajuste=-0.05, original=128.05)
    assert db.get_community_beat_grid(FP)['validated'] is True


def test_el_genero_no_se_aplica_con_dos_votos():
    # El bloque de consenso vive dentro de `analyze_audio` y no se puede
    # llamar sin librosa; se mira el fuente. Lo que se busca es que no quede
    # ninguna rama que aplique un voto de menos de tres.
    with open(os.path.join(AQUI, 'main.py'), encoding='utf-8') as fh:
        fuente = fh.read()
    assert 'suggestion_{' not in fuente, 'volvió el género con dos votos'
    # Solo el bloque que APLICA el consenso al análisis. `/correction` sigue
    # contestando «suggestion» con dos votos, pero eso es una etiqueta de la
    # respuesta: no cambia el tema de nadie.
    bloque = fuente[fuente.index('PRIORIDAD DE GENERO'):
                    fuente.index('Guardar en BD. engine_source')]
    assert 'MIN_CUENTAS_CAMBIO_MANUAL' in bloque
    for umbral in re.findall(r'\[1\] >= (\d+)|votes >= (\d+)', bloque):
        n = int(next(u for u in umbral if u))
        assert n >= 3, f'un cambio manual se aplica con {n} votos'
