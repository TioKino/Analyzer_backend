"""
Precisión del motor de análisis — BPM, tonalidad, energía y estructura.

POR QUÉ EXISTE (auditoría 2026-08-09): la suite tenía 506 tests verdes y corría
en 3 segundos porque NO TOCABA AUDIO. `precision_analyzer.py`,
`spectral_classifier.py` y `essentia_analyzer.py` estaban al 0 % de cobertura, y
`chunked_analyzer.py` al 15 %. O sea: lo que decide la calidad del producto
—que el BPM y la tonalidad salgan bien— era exactamente lo único sin probar.
Una regresión de precisión al tocar el analizador habría pasado inadvertida
hasta que un DJ mezclara con un beat-grid mal puesto.

CÓMO: se sintetiza audio con ground truth conocido (claves de bombo a un BPM
exacto, acordes de frecuencias exactas) y se comprueba lo que devuelve
`analyze_audio`, que es la función que corre en producción. Nada de mocks del
DSP: entra un WAV y sale un AnalysisResult, igual que en `/analyze`.

QUÉ NO SON: no validan librosa ni persiguen la última décima. Las tolerancias
son anchas a propósito. Lo que cazan es la regresión GORDA — que el BPM salga a
la mitad o al doble, que la tonalidad se vaya de tono, que la energía deje de
ordenar, que la estructura no vea una intro. Eso es lo que rompe el producto.

HERMÉTICOS: sin red. Se desactivan el detector de género (Discogs/MusicBrainz),
el artwork (iTunes/Deezer/Last.fm) y el auto-trigger de AudD.
"""

import math
import os
import tempfile

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

SR = 22050

# Frecuencias temperadas de la 4ª octava. Ground truth de los tests de tonalidad.
NOTE_HZ = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
}


@pytest.fixture(scope="module")
def dsp():
    """`main` con todas las salidas de red desactivadas."""
    os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("SYNC_DB_PATH", tempfile.mktemp(suffix=".db"))
    os.environ.setdefault("PREVIEWS_DIR", tempfile.mkdtemp())
    os.environ.setdefault("ARTWORK_CACHE_DIR", tempfile.mkdtemp())
    import main

    main.GENRE_DETECTOR_ENABLED = False   # Discogs + MusicBrainz
    main.ARTWORK_ENABLED = False          # iTunes + Deezer + Last.fm
    main.AUDD_AUTO_ENABLED = False        # AudD
    return main


def _wav(samples) -> str:
    path = tempfile.mktemp(suffix=".wav")
    soundfile.write(path, samples, SR)
    return path


def _click_track(bpm: float, secs: float = 16.0, amplitude: float = 0.8):
    """Bombos a intervalo exacto: el ground truth de BPM más limpio posible."""
    n = int(SR * secs)
    y = np.zeros(n, dtype=np.float32)
    step = int(SR * 60.0 / bpm)
    klen = int(SR * 0.05)
    t = np.arange(klen) / SR
    kick = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40) * amplitude).astype(np.float32)
    for i in range(0, n - klen, step):
        y[i:i + klen] += kick
    return y


def _chord(freqs, secs: float = 12.0):
    """Acorde sostenido con un armónico, para que el chroma tenga con qué."""
    t = np.arange(int(SR * secs)) / SR
    y = sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t)
            for f in freqs)
    return (y / np.max(np.abs(y)) * 0.7).astype(np.float32)


def _analyze(dsp, samples, name="Artista - Tema.wav"):
    return dsp.analyze_audio(_wav(samples), fingerprint="f" * 32,
                             original_filename=name)


# ==================== BPM ====================

class TestBPM:
    @pytest.mark.parametrize("bpm_real", [120, 128, 140])
    def test_detecta_el_tempo_de_un_patron_exacto(self, dsp, bpm_real):
        r = _analyze(dsp, _click_track(bpm_real))
        error = abs(r.bpm - bpm_real) / bpm_real
        assert error < 0.06, (
            f"BPM {r.bpm:.1f} para un patrón de {bpm_real} exactos "
            f"({error:.1%} de error)"
        )

    @pytest.mark.parametrize("bpm_real", [124, 132])
    def test_no_confunde_mitad_ni_doble(self, dsp, bpm_real):
        """El fallo clásico del beat-tracking, y el que más daño hace: un track
        de 128 detectado como 64 deja el beat-grid inservible para mezclar."""
        r = _analyze(dsp, _click_track(bpm_real))
        assert abs(r.bpm - bpm_real / 2) > 5, f"detectó la MITAD ({r.bpm:.1f})"
        assert abs(r.bpm - bpm_real * 2) > 5, f"detectó el DOBLE ({r.bpm:.1f})"

    def test_la_confianza_esta_en_rango(self, dsp):
        r = _analyze(dsp, _click_track(128))
        assert 0.0 <= r.bpm_confidence <= 1.0
        assert r.bpm_confidence > 0.5, (
            "un patrón perfectamente regular debería dar confianza alta; "
            f"dio {r.bpm_confidence:.2f}"
        )

    def test_el_bpm_es_siempre_positivo_y_musical(self, dsp):
        r = _analyze(dsp, _click_track(128))
        assert 40 < r.bpm < 250, f"BPM fuera de cualquier rango musical: {r.bpm}"


# ==================== TONALIDAD ====================

class TestTonalidad:
    @pytest.mark.parametrize("nombre,notas,esperado,camelot", [
        ("C mayor",  ['C', 'E', 'G'],    'C',   '8B'),
        ("G mayor",  ['G', 'B', 'D'],    'G',   '9B'),
        ("F# menor", ['F#', 'A', 'C#'],  'F#m', '11A'),
    ])
    def test_acorde_sintetico_da_su_tonalidad(self, dsp, nombre, notas, esperado, camelot):
        r = _analyze(dsp, _chord([NOTE_HZ[n] for n in notas]))
        assert r.key == esperado, f"{nombre}: esperaba {esperado}, dio {r.key}"
        assert r.camelot == camelot, f"{nombre}: camelot {camelot}, dio {r.camelot}"

    def test_distingue_mayor_de_menor(self, dsp):
        """La tríada menor no puede clasificarse como mayor: en Camelot eso
        manda al DJ a la rueda equivocada."""
        r = _analyze(dsp, _chord([NOTE_HZ['A'] / 2, NOTE_HZ['C'], NOTE_HZ['E']]))
        assert r.key is not None and r.key.endswith('m'), (
            f"un La menor debería salir menor; salió {r.key}"
        )

    def test_camelot_coherente_con_la_tabla(self, dsp):
        """El camelot no puede contradecir a la key: son el mismo dato."""
        from models import KEY_TO_CAMELOT

        r = _analyze(dsp, _chord([NOTE_HZ[n] for n in ('C', 'E', 'G')]))
        if r.key in KEY_TO_CAMELOT:
            assert r.camelot == KEY_TO_CAMELOT[r.key]


# ==================== ENERGÍA ====================

class TestEnergia:
    def test_ordena_fuerte_por_encima_de_flojo(self, dsp):
        """La energía es un valor relativo: lo que no puede pasar es que un
        track apagado puntúe por encima de uno que pega."""
        flojo = _analyze(dsp, _click_track(128, amplitude=0.05))
        fuerte = _analyze(dsp, _click_track(128, amplitude=0.95))
        assert fuerte.energy_dj >= flojo.energy_dj, (
            f"flojo={flojo.energy_dj} salió por encima de fuerte={fuerte.energy_dj}"
        )

    def test_esta_en_la_escala_1_10(self, dsp):
        r = _analyze(dsp, _click_track(128))
        assert 1 <= r.energy_dj <= 10, f"energy_dj fuera de escala: {r.energy_dj}"
        assert not math.isnan(r.energy_raw)


# ==================== ESTRUCTURA ====================

class TestEstructura:
    def test_ve_la_intro_y_el_outro_de_un_track_con_forma(self, dsp):
        """Silencio -> cuerpo -> silencio. Si esto no se detecta, los cue
        points automáticos y el track type salen de un análisis ciego."""
        suave = _click_track(128, secs=8, amplitude=0.05)
        fuerte = _click_track(128, secs=16, amplitude=0.95)
        y = np.concatenate([suave, fuerte, suave])
        r = _analyze(dsp, y)
        assert r.has_intro or r.has_outro, (
            "un track con entradas y salidas suaves debería marcar intro u outro"
        )
        assert len(r.structure_sections) > 0

    def test_las_secciones_van_en_orden_y_sin_huecos(self, dsp):
        r = _analyze(dsp, _click_track(128, secs=24))
        secciones = r.structure_sections
        for a, b in zip(secciones, secciones[1:]):
            assert a['start'] <= b['start'], "secciones desordenadas"
            assert a['end'] <= b['start'] + 0.01, "secciones solapadas"


# ==================== ROBUSTEZ ====================

class TestRobustez:
    def test_audio_vacio_no_revienta_con_un_error_opaco(self, dsp):
        """Un upload truncado decodifica a 0 muestras. Antes reventaba dentro
        de chroma_cqt con 'Input signal length=0 is too short for 7-octave CQT'
        tras gastar el ciclo entero de análisis."""
        with pytest.raises(ValueError, match="empty_audio_signal"):
            dsp.analyze_audio(_wav(np.zeros(0, dtype=np.float32)),
                              fingerprint="e" * 32,
                              original_filename="vacio.wav")

    def test_silencio_puro_no_lanza(self, dsp):
        """Silencio SÍ tiene muestras: debe analizarse sin excepción, aunque
        el resultado no signifique nada."""
        r = _analyze(dsp, np.zeros(int(SR * 5), dtype=np.float32))
        assert r.duration > 0
        assert r.bpm >= 0

    def test_es_determinista(self, dsp):
        """El mismo audio dos veces tiene que dar lo mismo. Sin esto, la
        memoria colectiva acumularía valores distintos para el mismo track."""
        y = _click_track(128)
        a, b = _analyze(dsp, y), _analyze(dsp, y)
        assert a.bpm == b.bpm
        assert a.key == b.key
        assert a.energy_dj == b.energy_dj

    def test_la_duracion_es_la_real(self, dsp):
        r = _analyze(dsp, _click_track(128, secs=18))
        assert abs(r.duration - 18.0) < 0.5, f"duración {r.duration:.1f}, esperaba 18"
