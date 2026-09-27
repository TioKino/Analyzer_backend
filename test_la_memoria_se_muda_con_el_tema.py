"""Lo que la comunidad deja en un tema no se pierde cuando ese tema entra en un
cluster acústico.

La clave de la memoria colectiva (`canonical_community_key`) es el cluster si
el tema lo tiene y su huella si no. Se decide al escribir y al leer, así que en
cuanto un tema recibe su huella acústica —el backfill, o la cura del
cache-hit— la clave CAMBIA, y todo lo escrito antes (notas, valoraciones,
votos, rejilla corregida) se quedaba debajo de la vieja: no lo veía nadie, ni
quien lo escribió. Medido el 2026-09-26 con la BD de verdad.

Hoy se muda en el momento en que el tema entra en el cluster
(`_mudar_memoria_al_cluster`) y lo huérfano de antes se recoge al arrancar
(`realinear_memoria_colectiva`).

    pytest test_la_memoria_se_muda_con_el_tema.py -v
"""

import uuid

import pytest

from database import AnalysisDB


@pytest.fixture
def db(tmp_path):
    return AnalysisDB(db_path=str(tmp_path / 'analysis.db'))


def _fp():
    return uuid.uuid4().hex


def _tema(db, fp, aid=None):
    db.conn.execute('INSERT INTO tracks (id, fingerprint, filename, acoustic_id) '
                    'VALUES (?, ?, ?, ?)', (fp, fp, fp + '.mp3', aid))
    db.conn.commit()


def _notas(db, fp):
    return [n['note_text'] for n in db.get_community_notes(fp)]


def test_la_nota_sigue_ahi_despues_del_backfill(db):
    fp = _fp()
    _tema(db, fp)
    db.save_community_note(fp, 'devA', 'entra en el 33')
    db.backfill_track_fingerprint(fp, 'Y2hyb21h', 'clusterX')
    assert _notas(db, fp) == ['entra en el 33']


def test_y_la_ve_quien_tiene_otro_fichero_del_mismo_tema(db):
    fp_mp3, fp_flac = _fp(), _fp()
    _tema(db, fp_mp3)
    db.save_community_note(fp_mp3, 'devA', 'ojo al break')
    db.backfill_track_fingerprint(fp_mp3, 'Y2hyb21h', 'clusterX')
    _tema(db, fp_flac, aid='clusterX')
    assert _notas(db, fp_flac) == ['ojo al break']


def test_tambien_las_valoraciones_y_la_popularidad(db):
    fp = _fp()
    _tema(db, fp)
    db.increment_popularity(fp, 'devA')
    db.rate_track(fp, 'devA', 4)
    db.backfill_track_fingerprint(fp, 'Y2hyb21h', 'clusterX')
    assert db.get_my_rating(fp, 'devA') == 4
    pop = db.get_track_popularity(fp)
    assert pop['analysis_count'] == 1
    assert pop['total_ratings'] == 1 and pop['avg_rating'] == 4


def test_la_popularidad_se_suma_a_la_del_cluster(db):
    fp_a, fp_b = _fp(), _fp()
    _tema(db, fp_b, aid='clusterX')
    db.increment_popularity(fp_b, 'devB')          # ya bajo el cluster
    _tema(db, fp_a)
    db.increment_popularity(fp_a, 'devA')          # aún bajo su huella
    db.backfill_track_fingerprint(fp_a, 'Y2hyb21h', 'clusterX')
    pop = db.get_track_popularity(fp_b)
    assert pop['analysis_count'] == 2
    assert pop['dj_count'] == 2


def test_la_rejilla_corregida_y_los_votos_tambien(db):
    fp = _fp()
    _tema(db, fp)
    db.submit_beat_grid_correction(fp, 'devA', 0.1, 0.01, 128.0)
    db.save_correction(fp, 'genre', 'House', 'Techno', fingerprint=fp,
                       device_id='devA')
    db.backfill_track_fingerprint(fp, 'Y2hyb21h', 'clusterX')
    fila = db.conn.execute(
        'SELECT COUNT(*) FROM beat_grid_corrections WHERE fingerprint = ?',
        ('clusterX',)).fetchone()[0]
    assert fila == 1
    assert db.conn.execute(
        'SELECT COUNT(*) FROM corrections WHERE fingerprint = ?',
        ('clusterX',)).fetchone()[0] == 1


def test_si_el_mismo_aparato_ya_voto_bajo_el_cluster_gana_el_nuevo(db):
    fp_a, fp_b = _fp(), _fp()
    _tema(db, fp_a)
    db.rate_track(fp_a, 'devA', 2)                 # viejo, bajo la huella
    _tema(db, fp_b, aid='clusterX')
    db.rate_track(fp_b, 'devA', 5)                 # nuevo, bajo el cluster
    db.backfill_track_fingerprint(fp_a, 'Y2hyb21h', 'clusterX')
    assert db.get_my_rating(fp_a, 'devA') == 5


def test_guardar_el_tema_con_cluster_tambien_muda(db):
    # La cura del cache-hit rellena el chromaprint re-guardando la fila.
    fp = _fp()
    _tema(db, fp)
    db.save_community_note(fp, 'devA', 'buen cierre')
    db.save_track({
        'id': fp, 'filename': fp + '.mp3', 'duration': 300.0, 'bpm': 128.0,
        'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
        'fingerprint': fp, 'acoustic_id': 'clusterX',
    })
    assert _notas(db, fp) == ['buen cierre']


def test_lo_huerfano_de_antes_se_recoge_al_arrancar(db):
    # Lo que se quedó bajo la huella ANTES de que existiera la mudanza: el
    # cluster se asignó por debajo, sin pasar por `backfill_track_fingerprint`.
    fp = _fp()
    _tema(db, fp)
    db.save_community_note(fp, 'devA', 'nota vieja')
    db.rate_track(fp, 'devA', 3)
    db.conn.execute("UPDATE tracks SET acoustic_id = 'clusterX' WHERE id = ?", (fp,))
    db.conn.commit()
    assert _notas(db, fp) == [], 'el caso de antes: huérfana'
    assert db.realinear_memoria_colectiva() >= 2
    assert _notas(db, fp) == ['nota vieja']
    assert db.get_my_rating(fp, 'devA') == 3
    assert db.realinear_memoria_colectiva() == 0, 'idempotente'


def test_lo_de_este_aparato_sale_con_la_huella_del_fichero(db):
    # El motor local guarda con SU cluster, que en Render no significa nada.
    fp = _fp()
    _tema(db, fp, aid='cluster-local-1')
    db.rate_track(fp, 'devA', 5)
    db.save_community_note(fp, 'devA', 'para el cierre')
    db.rate_track(_fp(), 'otro', 1)
    db.submit_community_override(fp, 'devA', 'genre', 'Techno')
    db.submit_beat_grid_correction(fp, 'devA', 0.1, 0.02, 128.0)
    r = db.lo_de_este_aparato('devA')
    assert r['ratings'] == [{'fingerprint': fp, 'rating': 5}]
    assert [(n['fingerprint'], n['note_text']) for n in r['notes']] == \
        [(fp, 'para el cierre')]
    assert r['overrides'] == [{'fingerprint': fp, 'field': 'genre', 'value': 'Techno'}]
    assert [(g['fingerprint'], g['bpm_adjust']) for g in r['rejillas']] == [(fp, 0.1)]


def test_lo_de_este_aparato_no_existe_en_render(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main, 'IS_LOCAL_ENGINE', False)
    r = TestClient(main.app).get('/community/de-este-aparato?device_id=x')
    assert r.status_code == 404
