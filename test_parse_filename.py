"""
Tests de parse_filename (audio_helpers).

Bug objetivo: la regex de extension solo cubria mp3|wav|flac|m4a, asi que los
otros formatos soportados (aac, ogg, aiff/aif, opus, wma) dejaban la extension
pegada al titulo ("Domino.aiff"), ensuciando lo mostrado y empeorando el match
en AudD/Discogs.

audio_helpers importa librosa a nivel modulo, asi que esto corre en CI (donde
librosa esta instalado), igual que el resto de la suite del backend.
"""

import pytest

from audio_helpers import parse_filename


@pytest.mark.parametrize("ext", [
    "mp3", "wav", "flac", "m4a", "aac", "ogg", "aiff", "aif", "opus", "wma",
])
def test_extension_stripped_for_all_supported_formats(ext):
    out = parse_filename(f"Oxia - Domino.{ext}")
    assert out["artist"] == "Oxia"
    assert out["title"] == "Domino"
    assert ext not in out["title"].lower()


def test_extension_case_insensitive():
    out = parse_filename("Oxia - Domino.FLAC")
    assert out["title"] == "Domino"


def test_artist_title_split():
    out = parse_filename("Charlotte de Witte - Sgadi Li Mi.mp3")
    assert out["artist"] == "Charlotte de Witte"
    assert out["title"] == "Sgadi Li Mi"


def test_no_separator_is_title_only():
    out = parse_filename("just_a_track.mp3")
    assert out["artist"] is None
    assert out["title"] == "just_a_track"


def test_leading_track_number_stripped():
    out = parse_filename("03 - Oxia - Domino.mp3")
    # El "03 - " inicial se quita; queda Artist - Title.
    assert out["artist"] == "Oxia"
    assert out["title"] == "Domino"


def test_whitespace_collapsed():
    out = parse_filename("Oxia   -   Domino.mp3")
    assert out["artist"] == "Oxia"
    assert out["title"] == "Domino"
