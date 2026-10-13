"""Volver a guardar una fila leída de la BD NO mete el análisis dentro de sí.

Hasta el 2026-10-13 el cache-hit por huella de `/analyze` (el mismo fichero
subido otra vez, con otro nombre) releía la fila con `_row_to_dict` —las
columnas, unos pocos campos del JSON y el `analysis_json` entero como texto—
y `save_track` lo volcaba tal cual. El JSON nuevo llevaba DENTRO el viejo,
escapado, y a cada reimportación otra capa:

- en primer nivel solo quedaban las columnas: la estructura, el croma y los
  demás rasgos se quedaban enterrados, y la ficha que se servía después iba
  sin ellos;
- la marca de un fallback (`"analysis_status": "failed"`) quedaba escapada,
  y el panel, que la busca como texto, contaba esa fila como «análisis OK sin
  huella» (las ráfagas de `macos-mas` de PENDING);
- y la fila crecía ~1,6× por reimportación (medido: 286 → 5.582 bytes en
  cinco).

    pytest test_el_analisis_no_se_anida.py -v
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from database import AnalysisDB, analisis_sin_anidar


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield AnalysisDB(db_path=path)
    try:
        os.unlink(path)
    except OSError:
        pass


# La fecha del análisis: vieja, para ver que reguardar no la mueve.
_ANALIZADO = (datetime.now() - timedelta(days=40)).isoformat()


def _analisis(**extra):
    """Un análisis como lo guarda `/analyze` la primera vez."""
    fila = {
        'id': 'fp1', 'fingerprint': 'fp1', 'filename': 'a.mp3',
        'artist': 'A', 'title': 'T', 'duration': 300.0, 'bpm': 128.0,
        'key': 'Am', 'camelot': '8A', 'energy_dj': 7, 'genre': 'Techno',
        'track_type': 'peak_time', 'chromaprint': 'AQAA' * 200,
        'acoustic_id': 'c1', 'platform': 'windows',
        'analyzed_at': _ANALIZADO,
        'structure_sections': [{'type': 'intro', 'start': 0, 'end': 30}],
        'croma': [0.1] * 12, 'groove_score': 0.7, 'cue_points': [1.0, 2.0],
    }
    fila.update(extra)
    return fila


def _fila(db, tid='fp1'):
    conn = sqlite3.connect(db.db_path)
    try:
        return conn.execute(
            'SELECT analysis_json, chromaprint, analyzed_at, platform, filename '
            'FROM tracks WHERE id = ?', (tid,)).fetchone()
    finally:
        conn.close()


def _reimportar(db, veces, tid='fp1'):
    """Lo que hace el cache-hit por huella de `/analyze`."""
    for i in range(veces):
        fila = db.get_track_by_fingerprint(tid)
        fila['filename'] = f'otro_{i}.mp3'
        db.save_track(fila)


def test_reimportar_no_hace_crecer_la_fila(db):
    db.save_track(_analisis())
    antes = len(_fila(db)[0])
    _reimportar(db, 6)
    despues = len(_fila(db)[0])
    # Las columnas que _row_to_dict añade (fecha, plataforma…) suman algo una
    # vez; lo que no puede es crecer con cada reimportación.
    assert despues < antes * 1.6
    _reimportar(db, 3)
    assert len(_fila(db)[0]) == despues


def test_los_rasgos_siguen_en_primer_nivel(db):
    db.save_track(_analisis())
    _reimportar(db, 3)
    aj = json.loads(_fila(db)[0])
    assert 'analysis_json' not in aj
    assert aj['structure_sections'] == [{'type': 'intro', 'start': 0, 'end': 30}]
    assert aj['croma'] == [0.1] * 12
    assert aj['groove_score'] == 0.7
    # Lo que cambió (el nombre del fichero) gana; la fecha y la huella siguen.
    assert aj['filename'] == 'otro_2.mp3'
    analysis_json, chromaprint, analyzed_at, platform, filename = _fila(db)
    assert chromaprint == 'AQAA' * 200
    assert platform == 'windows'
    assert filename == 'otro_2.mp3'
    assert analyzed_at == _ANALIZADO


def test_un_fallback_reguardado_sigue_siendo_un_fallback(db):
    """La marca la busca el panel como TEXTO: tiene que seguir en primer
    nivel, sin escapar, o la fila pasa a «análisis OK sin huella»."""
    db.save_track(_analisis(id='f1', fingerprint='f1', bpm=0.0, key=None,
                            chromaprint=None, acoustic_id=None,
                            analysis_status='failed',
                            analyzed_at=datetime.now().isoformat()))
    _reimportar(db, 2, tid='f1')
    assert '"analysis_status": "failed"' in _fila(db, 'f1')[0]
    hueco = db.acoustic_gap_breakdown()['by_outcome_last_7d']
    assert hueco['failed_fallback'] == 1
    assert hueco['analyzed_ok'] == 0


def test_un_analisis_nuevo_se_guarda_como_siempre():
    fila = _analisis()
    plano = analisis_sin_anidar(fila)
    assert plano == {k: v for k, v in fila.items() if k != 'chromaprint'}


def test_lo_de_fuera_manda_salvo_un_none():
    dentro = {'bpm': 128.0, 'isrc': 'ES1', 'structure_sections': [1]}
    fuera = {'bpm': 127.96, 'isrc': None, 'filename': 'b.mp3',
             'analysis_json': json.dumps(dentro)}
    plano = analisis_sin_anidar(fuera)
    assert plano['bpm'] == 127.96
    assert plano['isrc'] == 'ES1'
    assert plano['structure_sections'] == [1]
    assert plano['filename'] == 'b.mp3'


def test_un_json_roto_dentro_no_rompe_el_guardado():
    plano = analisis_sin_anidar({'bpm': 1.0, 'analysis_json': '{roto'})
    assert plano == {'bpm': 1.0}


def test_la_cura_aplana_lo_ya_anidado(db):
    """Lo que se guardó anidado antes del arreglo: al arrancar se aplana, sin
    tocar la fecha ni la huella, y una segunda pasada no hace nada."""
    db.save_track(_analisis())
    # Tres capas como las dejaba el save_track de antes.
    capa = json.loads(_fila(db)[0])
    for i in range(3):
        capa = {'id': 'fp1', 'filename': f'v{i}.mp3', 'bpm': 128.0,
                'analysis_json': json.dumps(capa)}
    conn = sqlite3.connect(db.db_path)
    conn.execute('UPDATE tracks SET analysis_json = ? WHERE id = ?',
                 (json.dumps(capa), 'fp1'))
    conn.commit()
    conn.close()

    assert db.aplanar_analisis_anidados() == 1
    aj = json.loads(_fila(db)[0])
    assert 'analysis_json' not in aj
    assert aj['croma'] == [0.1] * 12
    assert aj['filename'] == 'v2.mp3'
    assert _fila(db)[1] == 'AQAA' * 200
    assert _fila(db)[2] == _ANALIZADO
    assert db.aplanar_analisis_anidados() == 0


def test_render_la_cura_al_arrancar_y_lo_dice_siempre():
    src = Path(__file__).with_name('main.py').read_text(encoding='utf-8')
    i = src.index('def _aplanar_analisis_anidados():')
    tramo = src[i:i + 900]
    assert 'db.aplanar_analisis_anidados()' in tramo
    assert 'logger.info(f"[Tracks] {_n} análisis anidados aplanados")' in tramo
    assert "threading.Thread(target=_aplanar_analisis_anidados" in tramo
