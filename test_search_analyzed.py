"""Tests del endpoint /search-analyzed por ARTISTA+TITULO — el camino del que
depende el relleno de ghosts (GhostEnrichmentService consulta este endpoint por
nombre, no por huella). test_isrc.py cubria la via ISRC; esto cubre el fuzzy +
el stripping de sufijos de mezcla ("(Extended Mix)"), que es EXACTAMENTE lo que
hace funcionar el enriquecimiento de ghosts.
"""

import pytest
from fastapi.testclient import TestClient

from main import app, db as main_db

client = TestClient(app)


def _seed(fp, artist, title, bpm=128.0):
    main_db.save_track({
        'id': fp, 'filename': f'{fp}.mp3', 'artist': artist, 'title': title,
        'duration': 300.0, 'bpm': bpm, 'key': 'A min', 'camelot': '8A',
        'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
        'fingerprint': fp,
    })


def test_exact_artist_title_found():
    _seed('zzsa_1', 'ZZSearchArtist', 'ZZSearchTitle', bpm=131)
    r = client.get('/search-analyzed',
                   params={'artist': 'ZZSearchArtist', 'title': 'ZZSearchTitle'})
    assert r.status_code == 200
    body = r.json()
    assert body['found'] is True
    assert body['in_collective'] is True
    assert body['track']['bpm'] == 131


def test_strips_mix_suffix():
    # El ghost manda "Titulo (Extended Mix)"; debe casar con el "Titulo" guardado.
    _seed('zzsa_2', 'ZZMixArtist', 'ZZMixTitle', bpm=126)
    r = client.get('/search-analyzed', params={
        'artist': 'ZZMixArtist', 'title': 'ZZMixTitle (Extended Mix)'})
    assert r.json()['found'] is True


def test_case_insensitive():
    _seed('zzsa_3', 'ZZCaseArtist', 'ZZCaseTitle', bpm=124)
    r = client.get('/search-analyzed', params={
        'artist': 'zzcaseartist', 'title': 'zzcasetitle'})
    assert r.json()['found'] is True


def test_not_found_returns_false():
    r = client.get('/search-analyzed', params={
        'artist': 'ZZNoSuchArtistXYZ', 'title': 'ZZNoSuchTitleXYZ'})
    assert r.status_code == 200
    assert r.json()['found'] is False
    assert r.json()['track'] is None


def test_pending_bpm_zero_not_matched():
    # bpm=0 = 'pending' de /recognize, sin analisis real -> el endpoint lo ignora
    # (exige bpm>0). Asi un ghost no se rellena con un placeholder vacio.
    _seed('zzsa_4', 'ZZPendingArtist', 'ZZPendingTitle', bpm=0)
    r = client.get('/search-analyzed', params={
        'artist': 'ZZPendingArtist', 'title': 'ZZPendingTitle'})
    assert r.json()['found'] is False
