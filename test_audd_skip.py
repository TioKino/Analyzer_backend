"""
Tests del AHORRO de AudD por cluster-best: heredar identidad limpia del cluster
acustico en vez de gastar una llamada a AudD.

La idea: si otra copia del mismo SONIDO ya tiene artist/title limpios
(rekordbox / AudD previo de otro usuario), /analyze los hereda y NO llama a
AudD (el trigger de abajo ve metadata ya utilizable -> no dispara).

Testeamos las dos piezas aisladas del fpcalc:
  - db.cluster_identities: lectura de identidades del cluster.
  - _pick_clean_identity: elegir la primera NO basura.
"""

import os
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'skip.db'))


def _track(**over):
    t = {
        'id': 'x', 'filename': 'x.mp3', 'artist': 'Oxia', 'title': 'Domino',
        'duration': 400.0, 'bpm': 123.0, 'key': 'A min', 'camelot': '8A',
        'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
        'fingerprint': 'x', 'acoustic_id': 'CLUSTER1',
    }
    t.update(over)
    return t


# ── db.cluster_identities ────────────────────────────────────────────

def test_cluster_identities_returns_non_empty_pairs():
    db = _db()
    db.save_track(_track(id='a', fingerprint='a', artist='Oxia', title='Domino'))
    ids = db.cluster_identities('CLUSTER1')
    assert ('Oxia', 'Domino') in ids


def test_cluster_identities_skips_empty_fields():
    db = _db()
    db.save_track(_track(id='a', fingerprint='a', artist='', title='Domino'))
    db.save_track(_track(id='b', fingerprint='b', artist='Oxia', title='   '))
    assert db.cluster_identities('CLUSTER1') == []


def test_cluster_identities_empty_for_unknown_cluster():
    db = _db()
    db.save_track(_track())
    assert db.cluster_identities('OTRO') == []
    assert db.cluster_identities(None) == []


def test_cluster_identities_most_recent_first():
    import time
    db = _db()
    db.save_track(_track(id='old', fingerprint='old', artist='Old', title='One'))
    time.sleep(0.01)
    db.save_track(_track(id='new', fingerprint='new', artist='New', title='Two'))
    ids = db.cluster_identities('CLUSTER1')
    assert ids[0] == ('New', 'Two')


# ── _pick_clean_identity ─────────────────────────────────────────────

def test_pick_clean_identity_picks_first_non_garbage():
    from main import _pick_clean_identity
    # 'Unknown Artist' y 'Track 01' son basura; 'Oxia'/'Domino' no.
    ids = [('Unknown Artist', 'Track 01'), ('Oxia', 'Domino')]
    assert _pick_clean_identity(ids) == ('Oxia', 'Domino')


def test_pick_clean_identity_none_when_all_garbage():
    from main import _pick_clean_identity
    ids = [('Unknown', 'Untitled'), ('VA', '01'), ('', '')]
    assert _pick_clean_identity(ids) is None


def test_pick_clean_identity_empty_and_none():
    from main import _pick_clean_identity
    assert _pick_clean_identity([]) is None
    assert _pick_clean_identity(None) is None


def test_cluster_clean_identity_none_when_no_fingerprint(monkeypatch):
    """Best-effort: si fpcalc no da huella, devuelve None sin romper (cae a AudD
    como antes)."""
    import main
    monkeypatch.setattr(
        'acoustic_fingerprint.compute_raw_chromaprint', lambda *a, **k: None)
    assert main._cluster_clean_identity('/no/such.mp3', 400.0) is None
