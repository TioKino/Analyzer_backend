"""La energía y el tipo de tema que da el DSP, por camino y por duración
(#5 y #7 de PENDING, 2026-10-07).

Dos preguntas que no se podían contestar con datos:
- ¿Los temas largos de Render salían de verdad un nivel de energía arriba?
  El motor local analiza los largos por el camino corto, así que sus «más de
  5 min» son la comparación justa con los de trozos.
- ¿Cuánto *closing* sale, y dónde? Hasta el 2026-10-07 `classify_track_type`
  sumaba 1,0 a *closing* con un outro y más de 5 minutos; desde ese día el
  outro no cuenta, y lo ya analizado conserva lo que tenía.

    pytest test_rasgos_por_camino.py -v
"""

import json
import os
import uuid
from datetime import datetime, timedelta

import pytest

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _tema(db, *, duracion, energia, tipo, fuente='analysis', motor=None,
          outro=False, estado=None, bpm=128.0, hace_dias=0):
    fp = uuid.uuid4().hex
    fila = {
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3', 'bpm': bpm,
        'bpm_source': fuente, 'key': 'Am', 'camelot': '8A', 'key_source': fuente,
        'duration': duracion, 'energy_dj': energia, 'genre': 'Techno',
        'track_type': tipo, 'has_outro': outro, 'engine_source': motor,
        'analyzed_at': (datetime.utcnow() - timedelta(days=hace_dias)).isoformat(),
    }
    if estado:
        fila['analysis_status'] = estado
    db.save_track(fila)
    return fp


def test_reparte_por_camino_y_por_duracion(db):
    _tema(db, duracion=200, energia=5, tipo='opener')
    _tema(db, duracion=420, energia=6, tipo='closing', fuente='chunked_analysis',
          outro=True)
    _tema(db, duracion=420, energia=7, tipo='closing', fuente='chunked_analysis',
          outro=True)
    _tema(db, duracion=280, energia=6, tipo='warmup', fuente='chunked_analysis')
    # El motor local lo analiza todo por el corto, también lo largo.
    _tema(db, duracion=420, energia=5, tipo='peak_time', motor='local_engine')
    r = db.rasgos_por_camino(30)
    t = r['total']
    assert t['corto']['hasta_4min']['temas'] == 1
    largos = t['trozos']['mas_de_5min']
    assert largos['temas'] == 2
    assert largos['energia'] == {'6': 1, '7': 1}
    assert largos['energia_media'] == 6.5
    assert largos['tipos'] == {'closing': 2}
    assert 'con_outro' not in largos, 'el outro de antes salía de otro detector'
    assert r['recientes']['trozos']['mas_de_5min']['con_outro'] == 2
    assert t['trozos']['4_a_5min']['tipos'] == {'warmup': 1}
    assert t['motor_local']['mas_de_5min']['energia_media'] == 5
    assert r['recientes']['motor_local']['mas_de_5min']['con_outro'] == 0


def test_en_render_el_camino_lo_decide_la_duracion(db):
    """Es lo que decide Render (`CHUNK_ANALYSIS_THRESHOLD`), y no hace falta
    leer el JSON de cada fila para saberlo: también con BPM y tonalidad de
    las etiquetas."""
    _tema(db, duracion=400, energia=6, tipo='closing', fuente='id3')
    _tema(db, duracion=180, energia=4, tipo='warmup', fuente='id3')
    t = db.rasgos_por_camino(30)['total']
    assert t['trozos']['mas_de_5min']['temas'] == 1
    assert t['corto']['hasta_4min']['temas'] == 1


def test_fuera_lo_que_lleva_energia_y_tipo_de_relleno(db):
    """El fallback de un análisis fallido y las filas que siembra Escuchar
    llevan BPM 0 y energía y tipo de relleno: contarlas diría que el DSP da
    lo que nadie midió."""
    _tema(db, duracion=300, energia=5, tipo='peak_time', bpm=0, estado='failed')
    _tema(db, duracion=300, energia=5, tipo='peak_time', bpm=0,
          estado='recognize_only')
    assert db.rasgos_por_camino(30)['total'] == {}


def test_lo_viejo_no_es_reciente(db):
    _tema(db, duracion=420, energia=7, tipo='closing', fuente='chunked_analysis',
          hace_dias=90)
    _tema(db, duracion=420, energia=6, tipo='peak_time', fuente='chunked_analysis')
    r = db.rasgos_por_camino(30)
    assert r['total']['trozos']['mas_de_5min']['temas'] == 2
    assert r['recientes']['trozos']['mas_de_5min']['tipos'] == {'peak_time': 1}


def test_un_json_roto_no_tumba_la_medida(db):
    _tema(db, duracion=200, energia=5, tipo='opener')
    conn = db._open_conn()
    try:
        conn.execute("INSERT INTO tracks (id, fingerprint, filename, bpm, duration, "
                     "energy_dj, track_type, analysis_json, analyzed_at) "
                     "VALUES ('roto', 'roto', 'x.mp3', 128, 200, 5, 'opener', '{no', ?)",
                     (datetime.utcnow().isoformat(),))
        conn.commit()
    finally:
        conn.close()
    assert db.rasgos_por_camino(30)['total']['corto']['hasta_4min']['temas'] == 2


def test_el_outro_solo_lee_lo_reciente(db):
    """Leer `analysis_json` es lo caro (~2 KB por fila). El outro se cuenta
    solo en lo reciente y entrando por el índice de `analyzed_at`: por el de
    BPM, que es lo que SQLite elige si se le deja, recorre la tabla entera."""
    _tema(db, duracion=200, energia=5, tipo='opener')
    src = open('database.py', encoding='utf-8').read()
    i = src.index('def rasgos_por_camino')
    cuerpo = src[i:src.index('def lo_importado_de', i)]
    assert "WHERE analyzed_at >= ? AND +bpm > 0" in cuerpo
    assert "WHERE +bpm > 0 GROUP BY" in cuerpo
    conn = db._open_conn()
    try:
        plan = ' '.join(str(r[3]) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM tracks "
            "WHERE analyzed_at >= ? AND +bpm > 0 "
            "AND instr(analysis_json, 'has_outro') > 0", ('2026-01-01',)))
    finally:
        conn.close()
    assert 'idx_tracks_analyzed_at' in plan, plan


def test_sale_en_el_panel():
    import main  # noqa: F401  (monta el singleton que usa el panel)
    import routes.admin_panel as panel
    r = panel._rasgos_por_camino()
    assert r is not None, 'None = embudo.sh dice «el panel falló»'
    assert {'total', 'recientes', 'dias_recientes'} <= set(r)
    json.dumps(r)
    with open(panel.__file__, encoding='utf-8') as fh:
        assert '"rasgos_por_camino": _rasgos_por_camino()' in fh.read()
    embudo = os.path.join(os.path.dirname(__file__), '..', 'Analyzer',
                          'scripts', 'embudo.sh')
    if os.path.exists(embudo):
        with open(embudo, encoding='utf-8') as fh:
            assert "'rasgos_por_camino' not in t" in fh.read()
