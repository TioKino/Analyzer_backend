"""/recognize preprocesa como mucho un minuto, no el tema entero (2026-10-07).

El backfill de portadas del escritorio sube 6 MB de cada tema a /recognize
(origen=portada). El preprocesado pasaba el fichero ENTERO por `loudnorm`:
10-15 s de CPU por petición, y muchas veces se pasaba del tope de 15 s
(«ffmpeg normalize timeout» en el log de Render). Y de todo eso AudD solo
recibía 20 s: `_audd_clip_if_large` recorta 0:30→0:50.

Con el tope en la ENTRADA, ffmpeg lee un minuto, el WAV que sale (5,3 MB)
sigue pasando del umbral del recorte, y AudD recibe la misma ventana.

    pytest test_reconocer_no_normaliza_el_tema_entero.py -v
"""

import subprocess

import pytest

import main


@pytest.mark.parametrize('estrategia', ['normalize', 'aggressive', 'raw_wav'])
def test_ffmpeg_lee_como_mucho_un_minuto(monkeypatch, tmp_path, estrategia):
    comandos = []

    def correr(cmd, **_kw):
        comandos.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, b'', b'')

    monkeypatch.setattr(subprocess, 'run', correr)
    main._preprocess_audio_for_recognition(
        str(tmp_path / 'chunk.mp3'), str(tmp_path / 'out.wav'), estrategia)
    cmd = comandos[0]
    # El tope va DELANTE de `-i`: es de la entrada, así ffmpeg no lee el resto.
    i = cmd.index('-i')
    assert cmd[i - 2:i] == ['-t', str(main.SEGUNDOS_PARA_RECONOCER)]


def test_un_minuto_en_wav_sigue_pasando_por_el_recorte_de_audd():
    """Si el WAV de un minuto quedara por debajo del umbral de
    `_audd_clip_if_large`, AudD recibiría el minuto entero y no la ventana
    0:30→0:50 de siempre."""
    bytes_por_segundo = 44100 * 2  # mono, pcm_s16le, como sale de aquí
    wav = main.SEGUNDOS_PARA_RECONOCER * bytes_por_segundo
    assert wav > 4.0 * 1024 * 1024
    assert main.SEGUNDOS_PARA_RECONOCER >= 50, 'la ventana acaba en 0:50'
