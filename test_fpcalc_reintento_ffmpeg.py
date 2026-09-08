"""
Tests del reintento por ffmpeg cuando fpcalc no sabe decodificar el audio.

EL PORQUE. Medido en produccion el 2026-09-08: TODOS los fallos de huella
vivos eran `Error decoding audio frame`, sobre acapellas viejas —MP3 que llevan
veinte años rulando, con la cabecera hecha polvo—. Y `_attach_acoustic` solo
corre cuando el analisis ha ido bien, asi que en esos ficheros **librosa si
pudo leer el audio**: el unico que no podia era el decoder que fpcalc lleva
dentro. Eran huellas que se tiraban pudiendo sacarse, y con ellas los tracks se
quedaban fuera de la memoria colectiva.

El docstring de `compute_raw_chromaprint` ya describia este arreglo —«formato
que fpcalc no traga y el ffmpeg del bundle si (se transcodifica a WAV y se
reintenta)»— desde que se añadio la etiqueta del fichero al log. Estaba escrito
y no implementado: el codigo logueaba y hacia `return None`.
"""
import os
import shutil
import subprocess
import tempfile

import pytest

from acoustic_fingerprint import (
    _merece_transcodificar,
    compute_raw_chromaprint,
    fingerprints_match,
)


# ============================================================================
# La clasificacion — decide por MENSAJE, no por codigo de salida
# ============================================================================

class TestQueSeReintenta:
    """El codigo de salida de fpcalc NO separa las causas; el texto si.

    Esto costo una recomendacion equivocada el 2026-09-08: se habia clasificado
    `exit 3` como «no decodifica» y `exit 2` como «huella vacia», y los logs de
    produccion traian `exit 2` CON el mensaje de decodificacion. Si el reintento
    se hubiera enganchado al codigo de salida, se habria saltado justo parte de
    los ficheros que venia a rescatar.
    """

    @pytest.mark.parametrize('motivo', [
        "exit 3: ERROR: Error decoding audio frame (Invalid data found when processing input)",
        "exit 2: ERROR: Error decoding audio frame (Invalid data found when processing input)",
        "exit 1: ERROR: Invalid data found when processing input",
    ])
    def test_un_fallo_de_decodificacion_se_reintenta(self, motivo):
        assert _merece_transcodificar(motivo) is True

    def test_el_mismo_exit_2_con_otro_mensaje_NO_se_reintenta(self):
        # La pareja del test de arriba: mismo codigo de salida, causa opuesta.
        # `Empty fingerprint` es audio demasiado corto o silencio — fpcalc
        # decodifico perfectamente y no habia huella que sacar. Transcodificar
        # no alarga un fichero de dos segundos.
        assert _merece_transcodificar("exit 2: ERROR: Empty fingerprint") is False

    def test_un_timeout_NO_se_reintenta(self):
        # El fichero es largo o el disco va lento. Meter un transcode delante
        # solo lo empeora, y ademas gasta el doble de tiempo antes de rendirse.
        assert _merece_transcodificar("timeout (30s)") is False

    def test_sin_binario_NO_se_reintenta(self):
        # No es el fichero, es el entorno. Eso lo cubre `ensure_fpcalc`, que
        # reintenta pasado su propio plazo; aqui no hay nada que transcodificar.
        assert _merece_transcodificar("fpcalc no instalado") is False

    def test_exit_0_sin_huella_NO_se_reintenta(self):
        assert _merece_transcodificar("exit 0 pero sin fingerprint") is False

    def test_sin_motivo_no_hay_reintento(self):
        assert _merece_transcodificar(None) is False
        assert _merece_transcodificar("") is False


# ============================================================================
# El fichero temporal no se queda por ahi
# ============================================================================

def test_el_wav_temporal_se_borra_siempre(tmp_path, monkeypatch):
    """Corre por peticion y en un worker con disco acotado: un temporal por
    cada acapella rota llena `/tmp` sin que nadie lo relacione con esto."""
    import acoustic_fingerprint as af

    creados = []

    def falso_transcode(file_path, timeout):
        p = tmp_path / f'temp{len(creados)}.wav'
        p.write_bytes(b'RIFF fake')
        creados.append(p)
        return str(p)

    monkeypatch.setattr(af, 'ensure_fpcalc', lambda: '/bin/false')
    monkeypatch.setattr(af, '_transcodificar_a_wav', falso_transcode)
    # Primera pasada falla con decodificacion, segunda tambien.
    monkeypatch.setattr(
        af, '_fpcalc_una_pasada',
        lambda b, p, t: (None, 'exit 3: ERROR: Error decoding audio frame'),
    )

    assert af.compute_raw_chromaprint('/tmp/loquesea.mp3') is None
    assert creados, 'no se llego a transcodificar'
    for p in creados:
        assert not p.exists(), f'{p} se quedo sin borrar'


def test_el_wav_temporal_se_borra_tambien_cuando_rescata(tmp_path, monkeypatch):
    """El camino de exito es el que mas veces se recorre si esto funciona."""
    import acoustic_fingerprint as af

    creados = []

    def falso_transcode(file_path, timeout):
        p = tmp_path / 'ok.wav'
        p.write_bytes(b'RIFF fake')
        creados.append(p)
        return str(p)

    llamadas = []

    def falso_fpcalc(binario, path, timeout):
        llamadas.append(path)
        if len(llamadas) == 1:
            return None, 'exit 2: ERROR: Error decoding audio frame'
        return [1, 2, 3], None

    monkeypatch.setattr(af, 'ensure_fpcalc', lambda: '/bin/false')
    monkeypatch.setattr(af, '_transcodificar_a_wav', falso_transcode)
    monkeypatch.setattr(af, '_fpcalc_una_pasada', falso_fpcalc)

    assert af.compute_raw_chromaprint('/tmp/loquesea.mp3') == [1, 2, 3]
    assert len(llamadas) == 2, 'no reintento sobre el transcodificado'
    for p in creados:
        assert not p.exists(), f'{p} se quedo sin borrar'


def test_si_no_toca_reintentar_NO_se_transcodifica(monkeypatch):
    """Transcodificar cuesta CPU y disco. Un `Empty fingerprint` no debe
    pagarlo, porque no hay nada que ganar."""
    import acoustic_fingerprint as af

    monkeypatch.setattr(af, 'ensure_fpcalc', lambda: '/bin/false')
    monkeypatch.setattr(
        af, '_fpcalc_una_pasada',
        lambda b, p, t: (None, 'exit 2: ERROR: Empty fingerprint'),
    )

    def no_deberia(*a, **k):
        raise AssertionError('se transcodifico un fallo que no lo merecia')

    monkeypatch.setattr(af, '_transcodificar_a_wav', no_deberia)
    assert af.compute_raw_chromaprint('/tmp/corto.mp3') is None


# ============================================================================
# LO QUE DE VERDAD PUEDE SALIR MAL: que el transcode cambie la huella
# ============================================================================

_HAY_BINARIOS = shutil.which('ffmpeg') and shutil.which('fpcalc')


@pytest.mark.skipif(not _HAY_BINARIOS, reason='necesita ffmpeg + fpcalc')
def test_la_huella_del_transcodificado_es_EL_MISMO_audio():
    """La premisa entera del rescate, y la unica forma de que esto haga daño.

    Si pasar el audio por ffmpeg diera una huella distinta, estariamos
    sembrando clusters FALSOS: dos copias del mismo tema en cubos separados, y
    peor, un track rescatado que no casa con el mismo track de otro DJ. Eso es
    peor que no tener huella, porque rompe la promesa en silencio y ademas
    parece que funciona (la cobertura sube).

    Chromaprint esta diseñado justo para esto —el mismo audio en otro codec da
    el mismo cluster, es la premisa de la memoria colectiva— pero eso hay que
    demostrarlo aqui, no darlo por hecho.
    """
    with tempfile.TemporaryDirectory() as d:
        original = os.path.join(d, 'original.mp3')
        # 12 s con contenido variado: un tono plano da una huella degenerada y
        # el test pasaria sin probar nada. El clustering acota por duracion
        # (±2,5 s), asi que tambien hace falta que dure lo suficiente.
        subprocess.run(
            ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
             '-i', 'sine=frequency=440:duration=12',
             '-f', 'lavfi', '-i', 'anoisesrc=duration=12:amplitude=0.3',
             '-filter_complex', 'amix=inputs=2', original],
            check=True, capture_output=True,
        )

        del_original = compute_raw_chromaprint(original)
        assert del_original, 'fpcalc no saco huella del fichero de prueba'

        # El mismo audio, tal como lo deja el reintento: WAV mono 16 kHz.
        from acoustic_fingerprint import _transcodificar_a_wav
        wav = _transcodificar_a_wav(original, timeout=30)
        assert wav, 'ffmpeg no transcodifico'
        try:
            del_wav = compute_raw_chromaprint(wav)
        finally:
            os.unlink(wav)

        assert del_wav, 'fpcalc no saco huella del transcodificado'
        assert fingerprints_match(del_original, del_wav), (
            'la huella del transcodificado NO casa con la del original: el '
            'rescate sembraria clusters falsos'
        )


@pytest.mark.skipif(not _HAY_BINARIOS, reason='necesita ffmpeg + fpcalc')
def test_un_fichero_que_no_es_audio_no_rescata_nada():
    """El otro extremo: si el fichero esta roto de verdad, ffmpeg tampoco puede
    y la funcion devuelve None sin reventar `/analyze`. Best-effort de punta a
    punta."""
    with tempfile.TemporaryDirectory() as d:
        basura = os.path.join(d, 'roto.mp3')
        with open(basura, 'wb') as f:
            f.write(b'esto no es un mp3' * 100)
        assert compute_raw_chromaprint(basura) is None
