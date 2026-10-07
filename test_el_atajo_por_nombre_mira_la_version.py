"""
El atajo por NOMBRE de `/analyze` respeta `ANALYSIS_VERSION`.

Subir `ANALYSIS_VERSION` es la palanca para rehacer lo ya analizado cuando
cambia el DSP. El pre-check por huella y el cache-hit por huella ya miraban la
version, pero el atajo por nombre —mismo nombre Y misma huella, o sea el mismo
fichero subido otra vez, que es lo que hace todo el mundo al reimportar su
carpeta— devolvia la fila guardada sin mirarla. El cliente, avisado por el
pre-check de que su analisis era viejo, subia el fichero entero y recibia el
mismo analisis viejo.
"""

import hashlib
import io

import pytest


def _wav(freq, secs=6.0, sr=22050):
    import numpy as np
    import soundfile as sf

    t = np.arange(int(sr * secs)) / sr
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture()
def app_mod():
    import main

    main.GENRE_DETECTOR_ENABLED = False
    main.ARTWORK_ENABLED = False
    main.AUDD_AUTO_ENABLED = False
    return main


@pytest.fixture()
def client(app_mod):
    from fastapi.testclient import TestClient

    return TestClient(app_mod.app)


def _guardar(db, *, filename, fingerprint, version):
    db.save_track({
        'id': fingerprint,
        'filename': filename,
        'artist': 'DJ VIEJO',
        'title': 'Tema',
        'duration': 6.0,
        'bpm': 123.0,
        'key': 'Am',
        'camelot': '8A',
        'energy_dj': 7,
        'genre': 'Techno',
        'track_type': 'peak_time',
        'fingerprint': fingerprint,
        'analysis_version': version,
    })


def test_con_la_version_de_ahora_el_atajo_sirve(client, app_mod):
    audio = _wav(311.0)
    fp = hashlib.md5(audio).hexdigest()
    _guardar(app_mod.db, filename='mismo.mp3', fingerprint=fp,
             version=app_mod.ANALYSIS_VERSION)

    r = client.post('/analyze', files={'file': ('mismo.mp3', audio, 'audio/wav')})
    assert r.status_code == 200, r.text
    assert r.json().get('artist') == 'DJ VIEJO', (
        'el atajo legitimo dejo de funcionar: cada reimportacion reanalizaria')


def test_con_otra_version_se_analiza_otra_vez(client, app_mod, monkeypatch):
    audio = _wav(313.0)
    fp = hashlib.md5(audio).hexdigest()
    _guardar(app_mod.db, filename='viejo.mp3', fingerprint=fp, version='1')
    monkeypatch.setattr(app_mod, 'ANALYSIS_VERSION', '2')

    r = client.post('/analyze', files={'file': ('viejo.mp3', audio, 'audio/wav')})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d.get('artist') != 'DJ VIEJO' and d.get('bpm') != 123.0, (
        'el atajo por nombre devolvio el analisis de una version vieja: subir '
        'ANALYSIS_VERSION no rehace el caso mas comun')
