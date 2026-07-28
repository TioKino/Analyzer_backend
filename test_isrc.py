"""
Tests del ISRC como clave de IDENTIDAD exacta (columna tracks.isrc + lookup).

El ISRC (International Standard Recording Code) lo devuelve AudD en /recognize.
Guardarlo permite casar el MISMO tema entre usuarios/versiones sin depender del
fuzzy artist+title. Estos tests no tocan librosa/AudD: solo BD.
"""

import os
import tempfile

from database import AnalysisDB


def _db():
    return AnalysisDB(os.path.join(tempfile.mkdtemp(), 'isrc.db'))


def _track(**over):
    t = {
        'id': 'fp1', 'filename': 'a.mp3', 'artist': 'Oxia', 'title': 'Domino',
        'duration': 400.0, 'bpm': 123.0, 'key': 'A min', 'camelot': '8A',
        'energy_dj': 7, 'genre': 'Techno', 'track_type': 'peak_time',
        'fingerprint': 'fp1', 'isrc': 'GBABC1234567',
    }
    t.update(over)
    return t


def test_save_and_get_by_isrc():
    db = _db()
    db.save_track(_track())
    got = db.get_track_by_isrc('GBABC1234567')
    assert got is not None
    assert got['bpm'] == 123.0
    assert got['isrc'] == 'GBABC1234567'


def test_get_by_isrc_miss_returns_none():
    db = _db()
    db.save_track(_track())
    assert db.get_track_by_isrc('DOESNOTEXIST') is None


def test_get_by_isrc_none_arg_is_safe():
    db = _db()
    assert db.get_track_by_isrc(None) is None
    assert db.get_track_by_isrc('') is None


def test_get_by_isrc_prefers_most_recent():
    """Varias copias del mismo ISRC -> devuelve la analizada mas reciente."""
    import time
    db = _db()
    db.save_track(_track(id='old', fingerprint='old', bpm=120.0))
    time.sleep(0.01)  # analyzed_at se sella con datetime.now()
    db.save_track(_track(id='new', fingerprint='new', bpm=124.0))
    got = db.get_track_by_isrc('GBABC1234567')
    assert got is not None
    assert got['bpm'] == 124.0


def test_track_without_isrc_not_matched():
    """Un track guardado sin isrc no aparece al buscar por un isrc concreto."""
    db = _db()
    db.save_track(_track(isrc=None))
    assert db.get_track_by_isrc('GBABC1234567') is None
