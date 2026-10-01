"""
El ISRC que da AudD al identificar un FICHERO se guarda (2026-10-01).

En la prueba del owner, Shazam reconocía «Age of Love – The Age Of Love (Jam &
Spoon Watch Out For Stella Mix)» sonando SU fichero, pero el servidor no daba
ficha ni huellas: ese fichero estaba en Render como «The Age Of Love» a secas
y /analyze tiraba el ISRC de AudD (solo guardaba el de las etiquetas, que casi
nunca viene). Sin ISRC, nada une lo que Escuchar reconoce (Shazam siempre lo
da) con la huella del fichero.

Lo que se ata:
- `isrc_de_audd` lo saca de donde venga: arriba, apple_music, spotify,
  deezer o musicbrainz; normalizado y validado.
- /analyze lo guarda cuando AudD identifica el audio (las etiquetas mandan si
  traen uno).
- Con él, la ficha y las huellas se encuentran por ISRC aunque el título
  guardado no diga la versión.
"""
import os
import tempfile
import uuid

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

import audd_helper
import main
from audd_helper import isrc_de_audd
from routes.search import buscar_analizado

ISRC = 'DEA619100045'


class TestIsrcDeAudd:
    @pytest.mark.parametrize('datos', [
        {'isrc': ISRC},
        {'apple_music': {'isrc': ISRC}},
        {'spotify': {'external_ids': {'isrc': ISRC}}},
        {'deezer': {'isrc': ISRC}},
        {'musicbrainz': [{'isrcs': [ISRC]}]},
        {'isrc': 'de-a61-91-00045'},  # minúsculas y guiones
        {'isrc': 'basura', 'deezer': {'isrc': ISRC}},  # el primero válido
    ])
    def test_lo_saca_de_donde_venga(self, datos):
        assert isrc_de_audd(datos) == ISRC

    @pytest.mark.parametrize('datos', [
        None, {}, {'isrc': ''}, {'isrc': 'NOESUNISRC'}, {'spotify': 'x'},
        {'musicbrainz': 'x'},
    ])
    def test_sin_isrc_valido_none(self, datos):
        assert isrc_de_audd(datos) is None


@pytest.fixture
def analizar(monkeypatch):
    """`analyze_audio` sobre un patrón de bombos, con AudD de mentira."""
    monkeypatch.setattr(main, 'GENRE_DETECTOR_ENABLED', False)
    monkeypatch.setattr(main, 'ARTWORK_ENABLED', False)
    monkeypatch.setattr(main, 'AUDD_AUTO_ENABLED', True)
    monkeypatch.setattr(main, 'AUDD_API_TOKEN', 'x')
    monkeypatch.setattr(audd_helper, 'download_artwork_from_audd',
                        lambda *_a, **_k: None)

    def correr(audd, nombre='Age Of Love - The Age Of Love.wav'):
        monkeypatch.setattr(audd_helper, 'enrich_with_audd_if_needed',
                            lambda **_k: audd)
        sr = 22050
        y = np.zeros(sr * 12, dtype=np.float32)
        paso = int(sr * 60 / 132.5)
        t = np.arange(int(sr * 0.05)) / sr
        bombo = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40) * 0.8)
        for i in range(0, len(y) - len(bombo), paso):
            y[i:i + len(bombo)] += bombo
        ruta = tempfile.mktemp(suffix='.wav')
        soundfile.write(ruta, y, sr)
        try:
            return main.analyze_audio(ruta, fingerprint=uuid.uuid4().hex,
                                      original_filename=nombre)
        finally:
            os.remove(ruta)

    return correr


def test_analyze_guarda_el_isrc_que_da_audd(analizar):
    r = analizar({
        'artist': 'Age of Love',
        'title': 'The Age Of Love (Jam & Spoon Watch Out For Stella Mix)',
        'apple_music': {'isrc': ISRC},
    })
    assert r.isrc == ISRC
    assert r.title == 'The Age Of Love (Jam & Spoon Watch Out For Stella Mix)'


def test_sin_audd_no_hay_isrc_inventado(analizar):
    assert analizar(None).isrc is None


def test_con_el_isrc_la_ficha_y_la_huella_salen_aunque_el_titulo_no_diga_la_version():
    # El fichero del owner: guardado con el título sin versión, pero con el
    # ISRC que dio AudD al identificar su audio.
    fp = uuid.uuid4().hex
    s = uuid.uuid4().hex[:6]  # la BD es compartida con otros ficheros de test
    isrc = 'DEA61910' + f'{uuid.uuid4().int % 10000:04d}'
    main.db.save_track({
        'id': fp, 'fingerprint': fp, 'filename': f'{fp}.mp3',
        'artist': f'Age Of Love {s}', 'title': f'The Age Of Love {s}',
        'bpm': 132.5, 'duration': 420, 'key': 'Am', 'camelot': '8A',
        'energy_dj': 7, 'genre': 'Trance', 'track_type': 'peak_time',
        'isrc': isrc,
    })
    detectado = (f'Age of Love {s}',
                 f'The Age Of Love {s} (Jam & Spoon Watch Out For Stella Mix)')
    assert buscar_analizado(*detectado) is None, \
        'por nombre sigue sin casar: el título guardado no dice la versión'
    ficha = buscar_analizado(*detectado, isrc=isrc)
    assert ficha is not None and abs(ficha['bpm'] - 132.5) < 0.01
    assert fp in main.db.huellas_del_tema(ficha.get('fingerprint'), isrc), \
        'con la huella, el móvil dice «en mi biblioteca por el fichero»'
