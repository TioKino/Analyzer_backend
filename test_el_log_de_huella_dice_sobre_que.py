"""El log decia que fpcalc habia fallado, pero no sobre QUE.

Salio de un log real de Render (7 dias, 2026-09-07): tres
`[Acoustic] fpcalc exit 2` y ninguna forma de saber a que ficheros
correspondian. Y esa es justo la pregunta que decide, porque las dos
respuestas piden cosas OPUESTAS:

  fichero roto de verdad     -> no hay nada que arreglar, ese track se queda
                                fuera de la memoria colectiva y ya.
  formato que fpcalc no traga -> el ffmpeg del bundle si puede: se transcodifica
                                a WAV temporal y se reintenta. Se recuperan.

La trampa esta en QUE nombre se loguea. A `compute_raw_chromaprint` le llega el
temporal de la subida (`/tmp/tmpab12cd.mp3`), que no identifica nada. El nombre
util —el original mas la huella— lo tiene quien llama, en `track_data`. Por eso
la etiqueta se pasa desde fuera.

    pytest test_el_log_de_huella_dice_sobre_que.py -v
"""

import logging
import os
import subprocess
import tempfile

import pytest

import acoustic_fingerprint as af


@pytest.fixture
def fichero():
    fd, path = tempfile.mkstemp(suffix='.m4a')
    os.write(fd, b'no soy audio')
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class _Salida:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _con_fpcalc(monkeypatch, salida=None, excepcion=None):
    """Finge que fpcalc existe y devuelve `salida` (o lanza `excepcion`)."""
    monkeypatch.setattr(af, 'ensure_fpcalc', lambda: '/fake/fpcalc')

    def _run(*a, **k):
        if excepcion:
            raise excepcion
        return salida
    monkeypatch.setattr(af.subprocess, 'run', _run)


# ============================================================================
# EL NOMBRE SALE EN EL LOG
# ============================================================================

def test_el_exit_distinto_de_cero_dice_sobre_que_fichero(
        monkeypatch, caplog, fichero):
    _con_fpcalc(monkeypatch, _Salida(
        returncode=2,
        stderr='ERROR: Could not find any audio stream in the file'))
    with caplog.at_level(logging.WARNING):
        assert af.compute_raw_chromaprint(
            fichero, etiqueta='Radio Slave - Grindhouse.m4a [ab12cd34]') is None
    msg = caplog.text
    assert 'fpcalc exit 2' in msg
    assert 'Radio Slave - Grindhouse.m4a' in msg
    # Y sigue trayendo el motivo de fpcalc, que es lo que distingue «roto» de
    # «formato que no traga».
    assert 'Could not find any audio stream' in msg


def test_sin_etiqueta_cae_al_basename_que_al_menos_da_la_extension(
        monkeypatch, caplog, fichero):
    """El fallback no puede ser el path entero: en Render es un temporal con
    nombre aleatorio y ocupa la linea sin decir nada. El basename al menos
    trae la extension, que ya orienta."""
    _con_fpcalc(monkeypatch, _Salida(returncode=2, stderr='ERROR: x'))
    with caplog.at_level(logging.WARNING):
        af.compute_raw_chromaprint(fichero)
    assert os.path.basename(fichero) in caplog.text
    # El directorio NO: es ruido y en Render siempre es /tmp.
    assert os.path.dirname(fichero) not in caplog.text


def test_el_timeout_no_comparte_mensaje_con_los_demas_fallos(
        monkeypatch, caplog, fichero):
    """Un timeout se arregla subiendo el timeout; los demas fallos no. Antes
    caian los dos en el mismo `fpcalc fallo:` — dos causas con arreglos
    distintos compartiendo mensaje."""
    _con_fpcalc(monkeypatch, excepcion=subprocess.TimeoutExpired('fpcalc', 30))
    with caplog.at_level(logging.WARNING):
        assert af.compute_raw_chromaprint(
            fichero, etiqueta='largo.wav', timeout=30) is None
    assert 'timeout' in caplog.text.lower()
    assert 'largo.wav' in caplog.text


def test_exit_0_sin_huella_deja_rastro(monkeypatch, caplog, fichero):
    """El caso mudo: fpcalc dice que todo bien y no trae array. Se contaba como
    «sin huella» sin escribir una sola linea, asi que en el log no existia."""
    _con_fpcalc(monkeypatch, _Salida(returncode=0, stdout='{"duration": 1}'))
    with caplog.at_level(logging.WARNING):
        assert af.compute_raw_chromaprint(fichero, etiqueta='mudo.mp3') is None
    assert 'sin fingerprint' in caplog.text
    assert 'mudo.mp3' in caplog.text


def test_el_camino_bueno_no_loguea_nada(monkeypatch, caplog, fichero):
    _con_fpcalc(monkeypatch, _Salida(
        returncode=0, stdout='{"fingerprint": [1, 2, 3]}'))
    with caplog.at_level(logging.WARNING):
        assert af.compute_raw_chromaprint(fichero, etiqueta='ok.mp3') == [1, 2, 3]
    assert caplog.text.strip() == ''


# ============================================================================
# DE DONDE SALE LA ETIQUETA
# ============================================================================

def test_la_etiqueta_lleva_nombre_original_Y_huella():
    """Los dos hacen falta y por motivos distintos: el nombre para reconocerlo
    y saber la extension, la huella para llegar a la fila de `tracks` (es su
    clave primaria)."""
    from main import _etiqueta
    e = _etiqueta({'filename': 'track.flac',
                   'fingerprint': 'abcdef0123456789' * 2})
    assert 'track.flac' in e
    assert 'abcdef012345' in e
    # Recortada: la huella entera son 32 chars de ruido en cada linea.
    assert 'abcdef0123456789abcdef' not in e


def test_la_etiqueta_aguanta_una_fila_a_medias():
    """Se construye en el camino de un fallo, asi que no puede fallar ella."""
    from main import _etiqueta
    assert _etiqueta({}) == '?'
    assert _etiqueta(None) == '?'
    assert 'x.mp3' in _etiqueta({'filename': 'x.mp3'})


def test_attach_acoustic_le_pasa_la_etiqueta():
    """El bug seria silencioso: sin esto vuelve a loguear el temporal y nadie
    lo nota, porque la linea sigue saliendo."""
    aqui = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(aqui, 'main.py'), encoding='utf-8') as f:
        src = f.read()
    i = src.index('def _attach_acoustic')
    cuerpo = src[i:i + 1600]
    assert 'etiqueta=_etiqueta(track_data)' in cuerpo
