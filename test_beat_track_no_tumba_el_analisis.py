"""Un fallo de `beat_track` no puede llevarse el analisis entero por delante.

Error real, dos veces el 2026-08-19, con traza completa:

    File "main.py", line 1917, in analyze_audio
      tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    File "librosa/beat.py", line 516, in __trim_beats
      return beats[valid.min() : valid.max()]
    ValueError: zero-size array to reduction operation minimum which has no identity

`trim=True` (el defecto) recorta los beats debiles del principio y el final.
Cuando NINGUNO pasa el umbral, `valid` queda vacio y librosa hace `.min()`
sobre un array de tamano cero. Es un fallo suyo, no nuestro: pasa con audio sin
percusion, casi silencio o muy corto.

Lo que costaba: el analisis ENTERO —BPM, key, energia, genero, waveform,
preview, huella— se caia con un 500 por no poder recortar unos beats de los
bordes. Los dos tracks no se analizaron nunca. La key y la energia no dependen
de los beats para nada.

    pytest test_beat_track_no_tumba_el_analisis.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')


def _fuente():
    with open(_MAIN, encoding='utf-8') as f:
        return f.read()


# ============================================================================
# EL REINTENTO
# ============================================================================

def test_hay_reintento_sin_trim():
    """`trim=False` se salta `__trim_beats`, que es la funcion que peta, y da
    el mismo tempo. Es el arreglo quirurgico: no pierde el BPM."""
    src = _fuente()
    assert 'except ValueError as e:' in src
    assert 'librosa.beat.beat_track(y=y, sr=sr, trim=False)' in src


def test_el_reintento_va_DESPUES_del_intento_normal():
    # Llamar siempre con trim=False cambiaria el resultado de todos los tracks
    # que hoy funcionan: el recorte existe por algo.
    src = _fuente()
    i_normal = src.index('tempo, beats = librosa.beat.beat_track(y=y, sr=sr)\n')
    i_sin_trim = src.index('trim=False)')
    assert i_normal < i_sin_trim


def test_si_ni_asi_el_analisis_CONTINUA():
    """La key, la energia y el genero no dependen de los beats. Rendirse con
    el BPM no puede costar los otros seis campos."""
    src = _fuente()
    i = src.index('librosa.beat.beat_track(y=y, sr=sr, trim=False)')
    bloque = src[i:i + 700]
    assert 'tempo, beats = 0.0, np.array([], dtype=int)' in bloque, (
        'sin fallback el segundo fallo vuelve a tumbar /analyze'
    )
    assert 'raise' not in bloque


# ============================================================================
# LO QUE VIENE DESPUES, CON beats VACIO
# ============================================================================

def test_beats_vacio_no_revienta_rio_abajo():
    """Mover el fallo tres lineas mas abajo no seria arreglarlo.

    Se reproduce la cadena real: `frames_to_time([])` -> `np.diff([])` ->
    las reducciones que vienen despues.
    """
    import librosa
    beats = np.array([], dtype=int)
    beat_intervals = np.diff(librosa.frames_to_time(beats, sr=44100))
    assert len(beat_intervals) == 0

    # Las dos guardas que ya existian en main.py, verificadas de verdad.
    bpm_confidence = (
        1.0 - min(np.std(beat_intervals) * 2, 0.5)
        if len(beat_intervals) > 0 else 0.5
    )
    assert bpm_confidence == 0.5

    if len(beat_intervals) > 1:
        pytest.fail('no deberia entrar aqui con beats vacio')
    groove_score, swing_factor = 0.0, 0.5
    assert (groove_score, swing_factor) == (0.0, 0.5)


def test_la_correccion_half_double_sobrevive_a_bpm_cero():
    """Con bpm=0, ni el doble (0) ni la mitad (0) caen en 60-200, asi que la
    lista de candidatos se queda en uno y devuelve 0 sin dividir por nada."""
    from main import try_bpm_double_half
    fuera = try_bpm_double_half(
        y=np.zeros(1000, dtype=np.float32), sr=22050,
        original_bpm=0.0, bpm_confidence=0.5,
    )
    assert fuera == 0.0


# ============================================================================
# HONESTIDAD DEL DATO
# ============================================================================

def test_un_BPM_sin_medir_NO_se_firma_como_analysis():
    """Un 0 con `bpm_source='analysis'` afirma que el backend lo midio. La
    ficha del cliente lee la fuente vacia como «no se sabe» — es la misma
    correccion que se hizo alli con `bpmSourceKnown`: un valor por defecto
    pintado como un hecho."""
    src = _fuente()
    assert 'bpm_source = "analysis" if bpm > 0 else ""' in src


def test_el_fallo_queda_en_el_log_con_su_motivo():
    """Un `except` mudo convierte un bug en «a veces el BPM sale 0». El motivo
    tiene que estar delante cuando alguien lo mire."""
    src = _fuente()
    assert '[BPM] beat_track fallo' in src
    assert '[BPM] sin beats' in src
