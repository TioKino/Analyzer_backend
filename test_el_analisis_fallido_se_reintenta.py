"""
Un análisis FALLIDO se reintenta según POR QUÉ falló (2026-10-14).

Hasta ese día el fallback de `/analyze` (BPM 0, `analysis_status: 'failed'`)
era para siempre: el pre-check lo daba por analizado, la app ni volvía a subir
el fichero, y si lo subía el cache-hit le devolvía el fallo. Un tema que falló
por un error NUESTRO se quedaba a BPM 0 aunque el error se arreglara.

Hoy el fallback guarda el motivo y los intentos (`analisis_fallido`): un
fichero ilegible no se reintenta (está roto), cualquier otro fallo como mucho
tres veces, y uno de antes de esto, una vez, para saber por qué falló.
"""

import hashlib
import io
import subprocess

import pytest

import analisis_fallido as af


# ---------------------------------------------------------------- la regla

def _fallo(motivo=None, intentos=None):
    fila = {'analysis_status': 'failed', 'bpm': 0}
    if motivo is not None:
        fila['fallo'] = motivo
    if intentos is not None:
        fila['intentos_fallidos'] = intentos
    return fila


def test_un_fichero_ilegible_no_se_reintenta():
    assert af.se_reintenta(_fallo(af.ILEGIBLE, 1)) is False


def test_un_error_nuestro_se_reintenta_tres_veces():
    # `intentos` cuenta los que fallaron: el primero y tres reintentos.
    for intentos in (1, 2, 3):
        assert af.se_reintenta(_fallo(af.ERROR, intentos)), intentos
    assert af.se_reintenta(_fallo(af.ERROR, 4)) is False


def test_un_fallo_de_antes_se_reintenta_una_vez():
    assert af.se_reintenta(_fallo()) is True, (
        'sin motivo no se sabe si el fichero está roto: un intento lo dice')
    assert af.se_reintenta(_fallo(intentos=2)) is False


def test_lo_que_no_es_un_fallo_no_se_reintenta():
    assert af.se_reintenta({'bpm': 128.0, 'key': 'Am'}) is False
    assert af.se_reintenta({}) is False
    assert af.se_reintenta(None) is False
    assert af.fallo_de({'bpm': 128.0}) is None


def test_se_lee_del_json_guardado():
    import json
    fila = {'bpm': 0, 'analysis_json': json.dumps(
        {'analysis_status': 'failed', 'fallo': 'error', 'intentos_fallidos': 2})}
    assert af.fallo_de(fila) == {'motivo': 'error', 'intentos': 2}
    assert af.se_reintenta(fila)


def test_los_intentos_se_suman():
    assert af.intentos_tras(None) == 1
    assert af.intentos_tras({'bpm': 120.0}) == 1
    assert af.intentos_tras(_fallo(af.ERROR, 2)) == 3
    assert af.intentos_tras(_fallo()) == 2


# ------------------------------------------------------------ de verdad

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


class _Revienta:
    """Un `analyze_audio` que falla siempre, y cuenta las veces."""

    def __init__(self, error):
        self.error = error
        self.veces = 0

    def __call__(self, *a, **k):
        self.veces += 1
        raise self.error


def _fila(fp, filename, **extra):
    """Una fila como la que deja el fallback: BPM 0, sin tonalidad."""
    fila = {'id': fp, 'fingerprint': fp, 'filename': filename,
            'artist': 'X', 'title': 'Y', 'duration': 0, 'bpm': 0,
            'key': None, 'camelot': None, 'energy_dj': 5,
            'genre': 'Unknown', 'track_type': 'unknown'}
    fila.update(extra)
    return fila


def _subir(client, nombre, audio):
    r = client.post('/analyze', files={'file': (nombre, audio, 'audio/wav')})
    assert r.status_code == 200, r.text
    return r.json()


def test_un_error_nuestro_se_reintenta_hasta_tres_veces(client, app_mod, monkeypatch):
    audio = _wav(401.0)
    fp = hashlib.md5(audio).hexdigest()
    revienta = _Revienta(RuntimeError('un bug del DSP'))
    monkeypatch.setattr(app_mod, 'analyze_audio', revienta)

    for intento in (1, 2, 3, 4):
        assert _subir(client, 'error.mp3', audio)['bpm'] == 0
        fila = app_mod.db.get_track_by_fingerprint(fp)
        assert af.fallo_de(fila) == {'motivo': 'error', 'intentos': intento}
    assert revienta.veces == 4

    # Tres reintentos y se acabó: el quinto se sirve de la caché.
    _subir(client, 'error.mp3', audio)
    assert revienta.veces == 4, 'un fallo agotado se volvió a intentar'
    # Y por la huella (otro nombre), igual.
    _subir(client, 'otro_nombre.mp3', audio)
    assert revienta.veces == 4


def test_un_fichero_ilegible_no_se_vuelve_a_intentar(client, app_mod, monkeypatch):
    audio = _wav(403.0)
    fp = hashlib.md5(audio).hexdigest()
    revienta = _Revienta(subprocess.CalledProcessError(1, 'ffmpeg'))
    monkeypatch.setattr(app_mod, 'analyze_audio', revienta)

    _subir(client, 'roto.mp3', audio)
    assert af.fallo_de(app_mod.db.get_track_by_fingerprint(fp)) == {
        'motivo': 'ilegible', 'intentos': 1}
    _subir(client, 'roto.mp3', audio)
    _subir(client, 'roto_otro_nombre.mp3', audio)
    assert revienta.veces == 1, (
        'un fichero ilegible se volvió a analizar: subirlo otra vez no lo arregla')


def test_si_el_reintento_sale_bien_el_tema_queda_analizado(client, app_mod, monkeypatch):
    audio = _wav(405.0)
    fp = hashlib.md5(audio).hexdigest()
    real = app_mod.analyze_audio
    monkeypatch.setattr(app_mod, 'analyze_audio', _Revienta(MemoryError()))
    _subir(client, 'memoria.mp3', audio)
    assert af.se_reintenta(app_mod.db.get_track_by_fingerprint(fp))

    monkeypatch.setattr(app_mod, 'analyze_audio', real)
    d = _subir(client, 'memoria.mp3', audio)
    # Un tono puro no tiene pulso (BPM 0 también saliendo bien): lo que dice
    # que no es el fallback es la fuente, que el fallback deja en `pending`.
    assert d.get('bpm_source') != 'pending'
    assert af.fallo_de(app_mod.db.get_track_by_fingerprint(fp)) is None, (
        'el análisis bueno arrastró la marca del fallo')


def test_un_fallo_de_antes_se_reintenta_una_vez_y_apunta_el_motivo(
        client, app_mod, monkeypatch):
    audio = _wav(407.0)
    fp = hashlib.md5(audio).hexdigest()
    app_mod.db.save_track(_fila(fp, 'antiguo.mp3', analysis_status='failed'))
    revienta = _Revienta(subprocess.CalledProcessError(1, 'ffmpeg'))
    monkeypatch.setattr(app_mod, 'analyze_audio', revienta)

    _subir(client, 'antiguo.mp3', audio)
    assert revienta.veces == 1
    assert af.fallo_de(app_mod.db.get_track_by_fingerprint(fp)) == {
        'motivo': 'ilegible', 'intentos': 2}
    _subir(client, 'antiguo.mp3', audio)
    assert revienta.veces == 1


def test_el_pre_check_no_da_por_hecho_un_fallo_que_se_reintenta(client, app_mod):
    def guardar(nombre, **extra):
        fp = hashlib.md5(f'pre-check {nombre}'.encode()).hexdigest()
        app_mod.db.save_track(_fila(fp, f'{nombre}.mp3', **extra))
        return fp

    bueno = guardar('bueno', bpm=128.0, key='Am', camelot='8A', duration=300.0)
    ilegible = guardar('ilegible', analysis_status='failed', fallo='ilegible',
                       intentos_fallidos=1)
    agotado = guardar('agotado', analysis_status='failed', fallo='error',
                      intentos_fallidos=4)
    error = guardar('error', analysis_status='failed', fallo='error',
                    intentos_fallidos=1)
    antiguo = guardar('antiguo', analysis_status='failed')

    r = client.post('/check-analyzed-by-fingerprint', json={
        'fingerprints': [bueno, ilegible, agotado, error, antiguo]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d['analyzed']) == {bueno, ilegible, agotado}
    assert set(d['not_analyzed']) == {error, antiguo}, (
        'un fallo que se puede reintentar contó como analizado: la app no '
        'vuelve a subir el fichero y el tema se queda a BPM 0 para siempre')


def test_las_tres_puertas_miran_la_regla():
    from pathlib import Path
    main_src = Path('main.py').read_text(encoding='utf-8')
    assert main_src.count('analisis_fallido.se_reintenta(') == 2, (
        'los dos atajos de caché de /analyze (por nombre y por huella)')
    assert "track_data['fallo']" in main_src
    assert "track_data['intentos_fallidos']" in main_src
    rutas = Path('routes/analysis_artwork.py').read_text(encoding='utf-8')
    assert 'analisis_fallido.se_reintenta(' in rutas
