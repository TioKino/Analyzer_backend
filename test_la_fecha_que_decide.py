"""La fecha que decide si la via de la huella sigue abierta no puede mezclar
causas — y hasta hoy las mezclaba.

`newest_without` es, por diseño de esta doc, EL numero que separa «el hueco es
legado, lo cierra el backfill» de «hay algo activo produciendo tracks sin
huella». Pero se calculaba sobre TODAS las filas sin chromaprint, y dos de esas
causas salen sin huella A PROPOSITO:

  `failed`          el fallback de /analyze cuando librosa no puede con el
                    fichero. No llama a `_attach_acoustic` adrede.
  `recognize_only`  /recognize guarda su deteccion sobre un fragmento corto,
                    cuya huella descartaria el clustering por duracion.

O sea que una sola pulsacion de Escuchar ponia `newest_without` en HOY, y «la
via sigue abierta» se leia exactamente igual que «alguien uso Escuchar esta
tarde». Piden acciones opuestas: perseguir un bug, o nada.

Y ahora mismo es justo esa fecha lo que hay que leer. El 2026-09-06 se arreglo
que `ensure_fpcalc` memoizara el FALLO —la huella se apagaba entera hasta el
siguiente deploy, ~682 tracks de una tacada—, y esos 682 estan dentro de la
ventana de 30 dias: `analyzed_ok` seguira alto un mes aunque no entre ni uno
mas. La unica pregunta viva es si SIGUEN entrando, y solo la contesta una fecha
que excluya lo de diseño.

    pytest test_la_fecha_que_decide.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime, timedelta

import pytest

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(str(tmp_path / 'analysis.db'))


def _hace(dias):
    return (datetime.now() - timedelta(days=dias)).isoformat()


def _guardar(db, tid, *, cuando, status=None, chromaprint=None):
    datos = {
        'id': tid, 'filename': f'{tid}.mp3', 'duration': 300.0, 'bpm': 128.0,
        'energy_dj': 6, 'genre': 'Techno', 'track_type': 'peak',
        'fingerprint': tid, 'analyzed_at': cuando, 'chromaprint': chromaprint,
        'engine_source': 'render',
    }
    if status:
        datos['analysis_status'] = status
    db.save_track(datos)


def test_escuchar_hoy_no_puede_parecer_una_via_abierta(db):
    """El caso exacto: la unica fila reciente sin huella es un /recognize."""
    _guardar(db, 'viejo_ok', cuando=_hace(200))          # legado de verdad
    _guardar(db, 'escuchado', cuando=_hace(0), status='recognize_only')

    g = db.acoustic_gap_breakdown()

    assert g['newest_without'].startswith(datetime.now().strftime('%Y-%m-%d')), (
        'el maximo de siempre sigue viendo la fila de Escuchar'
    )
    assert g['newest_analyzed_ok_without'] == _o_fecha(db, 'viejo_ok'), (
        'pero la fecha que decide tiene que ignorarla: no hay via abierta'
    )


def test_un_fallback_de_analyze_tampoco(db):
    _guardar(db, 'viejo_ok2', cuando=_hace(120))
    _guardar(db, 'ilegible', cuando=_hace(0), status='failed')

    g = db.acoustic_gap_breakdown()
    assert not g['newest_analyzed_ok_without'].startswith(
        datetime.now().strftime('%Y-%m-%d'))


def test_un_analisis_OK_sin_huella_SI_mueve_la_fecha(db):
    """La otra mitad. Sin esto el arreglo escondería el bug que busca."""
    _guardar(db, 'viejo_ok3', cuando=_hace(120))
    _guardar(db, 'bug', cuando=_hace(0))          # sin marcador: paso por fpcalc

    g = db.acoustic_gap_breakdown()
    assert g['newest_analyzed_ok_without'].startswith(
        datetime.now().strftime('%Y-%m-%d')), 'aqui SI hay via abierta'


def test_una_fila_CON_huella_no_cuenta_para_nada(db):
    _guardar(db, 'sano', cuando=_hace(0), chromaprint='H')
    _guardar(db, 'viejo_ok4', cuando=_hace(300))

    g = db.acoustic_gap_breakdown()
    assert not g['newest_analyzed_ok_without'].startswith(
        datetime.now().strftime('%Y-%m-%d'))


def test_hay_reparto_a_7_dias_ademas_del_de_30(db):
    """Con solo la ventana de 30, un arreglo desplegado hoy no se puede
    comprobar hasta dentro de un mes: arrastra la rafaga anterior."""
    _guardar(db, 'rafaga_vieja', cuando=_hace(20))   # antes del arreglo
    _guardar(db, 'rafaga_vieja2', cuando=_hace(15))
    _guardar(db, 'reciente', cuando=_hace(1))        # despues

    g = db.acoustic_gap_breakdown()
    assert g['by_outcome_last_30d']['analyzed_ok'] == 3
    assert g['by_outcome_last_7d']['analyzed_ok'] == 1, (
        'la ventana corta es la que deja comprobar un arreglo de ayer'
    )


def test_el_reparto_de_7_dias_tambien_descuenta_lo_de_diseño(db):
    _guardar(db, 'escuchado7', cuando=_hace(1), status='recognize_only')
    _guardar(db, 'fallado7', cuando=_hace(1), status='failed')
    _guardar(db, 'bug7', cuando=_hace(1))

    o = db.acoustic_gap_breakdown()['by_outcome_last_7d']
    assert o == {'failed_fallback': 1, 'never_tried': 1, 'analyzed_ok': 1}


def test_sin_filas_no_revienta(db):
    g = db.acoustic_gap_breakdown()
    assert g['newest_analyzed_ok_without'] is None
    assert g['by_outcome_last_7d']['analyzed_ok'] == 0


def _o_fecha(db, tid):
    fila = db.get_track_by_fingerprint(tid)
    return fila['analyzed_at']
