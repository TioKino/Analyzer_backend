"""Regresion: ffmpeg abortaba la conversion por la CARATULA, no por el audio.

Sintoma en los logs de Render:

    [Preview] ffmpeg exit 1: Error while decoding stream #0:1: Invalid data
    found when processing input | Error marking filters as finished |
    Conversion failed!

Stream #0:1 de un mp3 no es audio: es el APIC del ID3 (la caratula) expuesto
como pista de video. Cuando la salida tambien es .mp3 —contenedor que admite
imagen incrustada— la seleccion automatica de ffmpeg coge "la mejor pista de
audio" Y "la mejor pista de video", decodifica la imagen para re-incrustarla
y, si esa imagen esta corrupta o truncada, tumba la conversion COMPLETA. El
audio estaba intacto: se perdia la preview por una caratula rota.

Afectaba a los dos sitios que producen .mp3:
  - generate_preview_snippet  → el snippet de 6s (error visible en el log)
  - _audd_clip_if_large       → el recorte de 20s para AudD (fallo MUDO: se
    enviaba el fichero entero y AudD respondia 413)

Los comandos que escriben .wav no estaban afectados: el muxer WAV no admite
video, asi que ffmpeg nunca selecciona la caratula.
"""

import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main  # noqa: E402


class _FakeCompleted:
    returncode = 0
    stdout = b''
    stderr = b''


def _index_of(cmd, flag):
    return cmd.index(flag) if flag in cmd else -1


class TestPreviewSnippet:

    @pytest.fixture
    def captured(self, monkeypatch, tmp_path):
        """Captura el cmd de ffmpeg sin ejecutarlo y simula un mp3 valido."""
        calls = []
        monkeypatch.setattr(main, 'PREVIEWS_DIR', str(tmp_path))

        real_exists = os.path.exists
        real_getsize = os.path.getsize
        out = str(tmp_path / 'fp123456.mp3')

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            with open(out, 'wb') as fh:
                fh.write(b'\x00' * 4096)  # > 1KB: el caller lo valida
            return _FakeCompleted()

        monkeypatch.setattr(subprocess, 'run', fake_run)
        # El snippet no debe existir antes de generarlo (si no, hace early
        # return), pero si despues.
        monkeypatch.setattr(os.path, 'exists',
                            lambda p: False if p == out and not calls else real_exists(p))
        monkeypatch.setattr(os.path, 'getsize', real_getsize)
        return calls

    def test_solo_toma_la_pista_de_audio(self, captured):
        main.generate_preview_snippet('/tmp/x.mp3', 'fp123456', 30.0, 300.0)
        assert captured, "ffmpeg no llego a invocarse"
        cmd = captured[0]
        assert '-map' in cmd
        assert cmd[_index_of(cmd, '-map') + 1] == '0:a:0'
        assert '-vn' in cmd

    def test_el_map_va_despues_del_input(self, captured):
        """'-map' es opcion de SALIDA: puesto antes de '-i' ffmpeg lo aplicaria
        al input equivocado o fallaria al parsear."""
        main.generate_preview_snippet('/tmp/x.mp3', 'fp123456', 30.0, 300.0)
        cmd = captured[0]
        assert _index_of(cmd, '-i') < _index_of(cmd, '-map')
        assert _index_of(cmd, '-i') < _index_of(cmd, '-vn')

    def test_sigue_generando_el_snippet_de_6s_mono(self, captured):
        """El fix no debe alterar el formato de salida."""
        main.generate_preview_snippet('/tmp/x.mp3', 'fp123456', 30.0, 300.0)
        cmd = captured[0]
        assert cmd[_index_of(cmd, '-t') + 1] == '6'
        assert cmd[_index_of(cmd, '-ac') + 1] == '1'
        assert cmd[_index_of(cmd, '-ar') + 1] == '22050'
        assert cmd[-1].endswith('.mp3')


class TestAuddClip:

    @pytest.fixture
    def big_file(self):
        fd, path = tempfile.mkstemp(suffix='.wav')
        with os.fdopen(fd, 'wb') as fh:
            fh.write(b'\x00' * (5 * 1024 * 1024))  # > 4MB: fuerza el recorte
        yield path
        for p in (path, f'{path}.audd20.mp3'):
            try:
                os.unlink(p)
            except OSError:
                pass

    def test_solo_toma_la_pista_de_audio(self, monkeypatch, big_file):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            with open(cmd[-1], 'wb') as fh:
                fh.write(b'\x00' * 1024)
            return _FakeCompleted()

        monkeypatch.setattr(subprocess, 'run', fake_run)
        out = main._audd_clip_if_large(big_file)

        assert out is not None, "no genero clip pese a superar el umbral"
        assert calls, "ffmpeg no llego a invocarse"
        cmd = calls[0]
        assert '-map' in cmd
        assert cmd[_index_of(cmd, '-map') + 1] == '0:a:0'
        assert '-vn' in cmd
        assert _index_of(cmd, '-i') < _index_of(cmd, '-map')

    def test_no_recorta_si_el_fichero_es_pequeno(self, monkeypatch):
        """Guardia de no-regresion: el fix no debe hacer que se recorte de mas."""
        fd, path = tempfile.mkstemp(suffix='.wav')
        with os.fdopen(fd, 'wb') as fh:
            fh.write(b'\x00' * 1024)
        try:
            monkeypatch.setattr(
                subprocess, 'run',
                lambda *a, **k: pytest.fail('no deberia invocar ffmpeg'))
            assert main._audd_clip_if_large(path) is None
        finally:
            os.unlink(path)
