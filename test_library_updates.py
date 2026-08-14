"""
MEMORIA COLECTIVA — "que gano MI biblioteca" (FASE 4 retencion).

Cubre get_community_updates_for_library: conteo de tracks propios que
recibieron aportaciones de OTROS DJs, exclusion del propio device, filtro
temporal `since` y agregacion por CLUSTER acustico (mismo audio, otro codec).
"""

import os
import tempfile
from datetime import datetime, timedelta

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'libupd.db'))


def _iso(days_ago=0):
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat()


def _add_cue(db, fingerprint, device_id, when, pos=1000):
    conn = db._open_conn()
    conn.execute(
        'INSERT OR REPLACE INTO community_cues '
        '(fingerprint, device_id, cue_type, position_ms, note, created_at) '
        'VALUES (?,?,?,?,?,?)',
        (fingerprint, device_id, 'hot', pos, 'x', when))
    conn.commit()
    conn.close()


def _add_rating(db, fingerprint, device_id, when, rating=5):
    conn = db._open_conn()
    conn.execute(
        'INSERT OR REPLACE INTO track_ratings '
        '(fingerprint, device_id, rating, rated_at) VALUES (?,?,?,?)',
        (fingerprint, device_id, rating, when))
    conn.commit()
    conn.close()


def test_biblioteca_vacia_devuelve_cero():
    db = _db()
    out = db.get_community_updates_for_library([])
    assert out['tracks_updated'] == 0
    assert out['examples'] == []


def test_cuenta_aportacion_de_otro_dj():
    db = _db()
    _add_cue(db, 'fpA', 'otroDJ', _iso(1))
    out = db.get_community_updates_for_library(['fpA'])
    assert out['tracks_updated'] == 1
    assert out['by_type'].get('cues') == 1


def test_excluye_mis_propias_aportaciones():
    """Contar lo que aporto YO como 'mejora de la comunidad' seria mentira."""
    db = _db()
    _add_cue(db, 'fpA', 'yo', _iso(1))
    out = db.get_community_updates_for_library(['fpA'], exclude_device_id='yo')
    assert out['tracks_updated'] == 0
    # Pero si lo aporta otro, SI cuenta.
    _add_cue(db, 'fpA', 'otroDJ', _iso(1), pos=2000)
    out = db.get_community_updates_for_library(['fpA'], exclude_device_id='yo')
    assert out['tracks_updated'] == 1


def test_filtro_since_ignora_lo_viejo():
    db = _db()
    _add_cue(db, 'fpA', 'otroDJ', _iso(10))       # viejo
    out = db.get_community_updates_for_library(['fpA'], since=_iso(5))
    assert out['tracks_updated'] == 0
    _add_cue(db, 'fpA', 'otroDJ', _iso(1), pos=2000)  # reciente
    out = db.get_community_updates_for_library(['fpA'], since=_iso(5))
    assert out['tracks_updated'] == 1


def test_solo_cuenta_mis_tracks():
    db = _db()
    _add_cue(db, 'fpMio', 'otroDJ', _iso(1))
    _add_cue(db, 'fpAjeno', 'otroDJ', _iso(1))
    out = db.get_community_updates_for_library(['fpMio'])
    assert out['tracks_updated'] == 1


def test_varias_fuentes_no_duplican_el_track():
    """Un mismo track con cue + rating cuenta UNA vez en tracks_updated,
    pero aparece en los dos by_type."""
    db = _db()
    _add_cue(db, 'fpA', 'otroDJ', _iso(1))
    _add_rating(db, 'fpA', 'otroDJ2', _iso(1))
    out = db.get_community_updates_for_library(['fpA'])
    assert out['tracks_updated'] == 1
    assert out['by_type'].get('cues') == 1
    assert out['by_type'].get('ratings') == 1


def test_agrega_por_cluster_acustico():
    """Otro DJ pone cues en SU copia (otro codec -> otro fingerprint). Como
    comparten cluster acustico, debe contar para MI copia."""
    db = _db()
    conn = db._open_conn()
    conn.execute(
        "INSERT INTO tracks (id, fingerprint, filename, acoustic_id) "
        "VALUES (?,?,?,?)", ('id1', 'fpMio', 'a.mp3', 'CLUSTER1'))
    conn.execute(
        "INSERT INTO tracks (id, fingerprint, filename, acoustic_id) "
        "VALUES (?,?,?,?)", ('id2', 'fpOtraCopia', 'a.flac', 'CLUSTER1'))
    conn.commit()
    conn.close()
    _add_cue(db, 'fpOtraCopia', 'otroDJ', _iso(1))
    out = db.get_community_updates_for_library(['fpMio'])
    assert out['tracks_updated'] == 1, 'el cluster debe agregar la otra copia'


def test_examples_traen_artista_y_titulo():
    db = _db()
    conn = db._open_conn()
    conn.execute(
        "INSERT INTO tracks (id, fingerprint, filename, artist, title) "
        "VALUES (?,?,?,?,?)", ('id1', 'fpA', 'a.mp3', 'Oxia', 'Domino'))
    conn.commit()
    conn.close()
    _add_cue(db, 'fpA', 'otroDJ', _iso(1))
    out = db.get_community_updates_for_library(['fpA'])
    assert out['tracks_updated'] == 1
    assert out['examples'] and out['examples'][0]['artist'] == 'Oxia'
    assert out['examples'][0]['title'] == 'Domino'


# ── Integracion HTTP: el endpoint debe estar MONTADO ─────────────────────
# En este proyecto los bugs de WIRING (router no registrado -> 404 silencioso)
# han sido los mas graves historicamente. Este test lo cubre.

def test_endpoint_montado_y_shape_correcto():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    r = client.post('/community/library-updates',
                    json={'fingerprints': ['fpQueNoExiste'],
                          'device_id': 'devX'})
    assert r.status_code == 200, 'router no registrado (404) o error'
    body = r.json()
    assert 'tracks_updated' in body
    assert body['tracks_updated'] == 0


def test_endpoint_acepta_lista_vacia():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    r = client.post('/community/library-updates', json={'fingerprints': []})
    assert r.status_code == 200
    assert r.json()['tracks_updated'] == 0
