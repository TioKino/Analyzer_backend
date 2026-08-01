"""Tests del ENDPOINT POST /backfill-fingerprint (camino de produccion del
boton "Mejorar mi biblioteca").

test_acoustic_clustering.py ya cubre el metodo de BD `backfill_track_fingerprint`
en aislamiento; esto cubre el endpoint HTTP completo: decodificar el chromaprint
del cliente -> resolver/crear cluster -> escribir al track -> devolver la mejor
metadata del cluster. Incluye el caso clave (dos copias del mismo audio comparten
cluster tras el backfill) y el guard de chromaprint invalido (nunca 500).
"""

import random

from fastapi.testclient import TestClient

from acoustic_fingerprint import encode_raw
from main import app, db as main_db

client = TestClient(app)


def _rand_fp(n=400, seed=1):
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(n)]


def _flip_bits(fp, n_flips, seed=99):
    rng = random.Random(seed)
    out = list(fp)
    total = len(fp) * 32
    for _ in range(n_flips):
        pos = rng.randrange(total)
        out[pos // 32] ^= (1 << (pos % 32))
    return out


def _seed_without_chromaprint(fp, *, bpm=128.0, duration=300.0):
    """Track YA analizado pero SIN huella (pre-fpcalc), como la biblioteca vieja."""
    main_db.save_track({
        'id': fp, 'filename': f'{fp}.mp3', 'artist': 'A', 'title': 'T',
        'duration': duration, 'bpm': bpm, 'key': 'A min', 'camelot': '8A',
        'energy_dj': 6, 'genre': 'Techno', 'track_type': 'peak',
        'fingerprint': fp,
    })


def test_invalid_chromaprint_returns_ok_false_no_500():
    r = client.post('/backfill-fingerprint', json={
        'fingerprint': 'whatever', 'chromaprint': 'not-base64-!!!',
    })
    assert r.status_code == 200          # nunca 500 (best-effort)
    body = r.json()
    assert body['ok'] is False
    assert body['reason'] == 'chromaprint_invalido'


def test_backfill_writes_cluster_and_returns_best():
    raw = _rand_fp(seed=11)
    _seed_without_chromaprint('bfapi_1', bpm=129.0)
    r = client.post('/backfill-fingerprint', json={
        'fingerprint': 'bfapi_1', 'chromaprint': encode_raw(raw), 'duration': 300.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] is True
    assert body['updated'] is True
    assert body['acoustic_id']            # se le asigno cluster
    # La huella + cluster quedaron escritos en el track existente.
    row = main_db.get_track_by_fingerprint('bfapi_1')
    assert row['chromaprint'] == encode_raw(raw)
    assert row['acoustic_id'] == body['acoustic_id']
    # Devuelve la mejor metadata del cluster (aqui, la del propio track).
    assert body['best'] is not None
    assert body['best']['bpm'] == 129.0


def test_two_copies_share_cluster_after_backfill():
    """El nucleo del backfill: dos copias del MISMO audio (otro codec) acaban en
    el MISMO cluster -> comparten memoria colectiva sin re-analizar."""
    original = _rand_fp(seed=22)
    reencoded = _flip_bits(original, 10)   # mismo audio, otro codec

    _seed_without_chromaprint('bfapi_orig', duration=310.0)
    _seed_without_chromaprint('bfapi_copy', duration=310.0)

    r1 = client.post('/backfill-fingerprint', json={
        'fingerprint': 'bfapi_orig', 'chromaprint': encode_raw(original),
        'duration': 310.0,
    })
    r2 = client.post('/backfill-fingerprint', json={
        'fingerprint': 'bfapi_copy', 'chromaprint': encode_raw(reencoded),
        'duration': 310.0,
    })
    aid1 = r1.json()['acoustic_id']
    aid2 = r2.json()['acoustic_id']
    assert aid1 and aid1 == aid2           # mismo cluster pese a otro chromaprint


def test_backfill_unknown_fingerprint_ok_but_not_updated():
    # chromaprint valido pero el track no existe -> se resuelve cluster pero
    # no hay fila que actualizar. No es error.
    raw = _rand_fp(seed=33)
    r = client.post('/backfill-fingerprint', json={
        'fingerprint': 'bfapi_ghost_missing', 'chromaprint': encode_raw(raw),
        'duration': 300.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] is True
    assert body['updated'] is False
