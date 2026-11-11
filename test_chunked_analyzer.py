"""
ChunkedAudioAnalyzer — la ruta de los tracks > 4 min.

POR QUÉ IMPORTA: en Render, todo track de más de 4 minutos va por aquí
(`main.CHUNK_ANALYSIS_THRESHOLD`), o sea buena parte de lo que analiza un DJ.
El módulo estaba al 15 % de cobertura y lo único probado era
`fuse_bpm_results` (la media de los BPM de cada trozo; desde el 2026-10-06 el
BPM sale como en el camino corto y esa función ya no existe — ver
`test_el_chunked_calcula_lo_mismo.py`). Toda la fusión de tonalidad, la
curva de energía, la estructura y el beat-grid iban sin red.

ESTRATEGIA: las funciones de fusión son puras (reciben listas de dicts), así que
se prueban con datos sintéticos sin generar audio. Los `analyze_chunk_*` toman
arrays directamente. Y hay UN test end-to-end de `full_analysis` con
`chunk_duration` reducido, que recorre el camino multi-chunk real sin necesidad
de sintetizar cinco minutos.

Varios tests fijan regresiones que YA pasaron en producción; están marcados.
"""

import math
import tempfile

import numpy as np
import pytest

pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

from chunked_analyzer import ChunkedAudioAnalyzer  # noqa: E402
from rasgos_del_tema import nivel_de_energia, rejilla_y_bpm  # noqa: E402

SR = 22050


@pytest.fixture(scope="module")
def analyzer():
    return ChunkedAudioAnalyzer(chunk_duration=15, chunk_overlap=2, sample_rate=SR)


def _click(bpm, secs, amp=0.8):
    n = int(SR * secs)
    y = np.zeros(n, dtype=np.float32)
    step = int(SR * 60.0 / bpm)
    klen = int(SR * 0.05)
    t = np.arange(klen) / SR
    kick = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40) * amp).astype(np.float32)
    for i in range(0, n - klen, step):
        y[i:i + klen] += kick
    return y


# La FUSIÓN DE TONALIDAD (un voto por trozo) se quitó el 2026-11-10: el de
# trozos cose la CQT del tema entero y saca la tonalidad con la misma cuenta
# que el corto (`test_la_tonalidad_log_braw.py`).


# ==================== CURVA DE ENERGÍA Y ESTRUCTURA ====================

class TestEnergiaYEstructura:
    def test_la_curva_sale_ordenada_en_el_tiempo(self, analyzer):
        """Los chunks se procesan en paralelo/desorden; la curva combinada
        tiene que quedar cronológica o la estructura se calcula sobre ruido."""
        chunks = [
            {'energy_curve': [{'time': 30.0, 'energy': 0.4}]},
            {'energy_curve': [{'time': 0.0, 'energy': 0.1}]},
            {'energy_curve': [{'time': 15.0, 'energy': 0.9}]},
        ]
        curva = analyzer.build_energy_curve(chunks)
        assert [p['time'] for p in curva] == [0.0, 15.0, 30.0]

    def test_curva_vacia_no_rompe_la_estructura(self, analyzer):
        r = analyzer.detect_structure_from_energy([], duration=300.0)
        assert r['sections'] == []
        assert r['has_drop'] is False
        assert r['drop_timestamp'] == pytest.approx(100.0)

    def test_ve_intro_y_outro_en_un_track_con_forma(self, analyzer):
        """Flojo al principio y al final, fuerte en medio."""
        curva = (
            [{'time': float(t), 'energy': 0.05} for t in range(0, 30, 5)]
            + [{'time': float(t), 'energy': 0.90} for t in range(30, 150, 5)]
            + [{'time': float(t), 'energy': 0.05} for t in range(150, 180, 5)]
        )
        r = analyzer.detect_structure_from_energy(curva, duration=180.0)
        assert r['has_intro'] or r['has_outro']
        assert len(r['sections']) > 0

    def test_el_drop_cae_dentro_del_track(self, analyzer):
        curva = [{'time': float(t), 'energy': 0.2 if t < 60 else 0.95}
                 for t in range(0, 240, 5)]
        r = analyzer.detect_structure_from_energy(curva, duration=240.0)
        assert 0 <= r['drop_timestamp'] <= 240.0


# ==================== ESCALA DE ENERGÍA ====================

class TestEscalaDeEnergia:
    @pytest.mark.parametrize("raw,esperado", [(0.0, 1), (0.01, 1), (0.42, 10), (1.0, 10)])
    def test_los_extremos_se_saturan(self, analyzer, raw, esperado):
        assert nivel_de_energia(raw) == esperado

    def test_siempre_dentro_de_1_10(self, analyzer):
        for raw in (0.0, 0.05, 0.1, 0.2, 0.3, 0.41, 0.5, 5.0):
            v = nivel_de_energia(raw)
            assert 1 <= v <= 10, f"raw={raw} -> {v}"

    def test_es_monotona(self, analyzer):
        vals = [nivel_de_energia(r) for r in
                (0.03, 0.08, 0.15, 0.25, 0.35, 0.41)]
        assert vals == sorted(vals), f"la escala no es monótona: {vals}"

    @pytest.mark.parametrize("malo", [float('nan'), float('inf'), float('-inf')])
    def test_nan_e_inf_no_revientan(self, analyzer, malo):
        """REGRESIÓN: era el error #1 del panel admin (112 ocurrencias). Un RMS
        NaN hacía que las comparaciones dieran False y `int(NaN)` explotara con
        ValueError, tumbando el análisis entero."""
        v = nivel_de_energia(malo)
        assert 1 <= v <= 10 and not math.isnan(v)


# ==================== BEAT GRID ====================

class TestBeatGrid:
    """La rejilla de los dos caminos es `rasgos_del_tema.rejilla_y_bpm`."""

    def test_el_intervalo_es_60_partido_bpm(self):
        fb, iv, bpm = rejilla_y_bpm(np.zeros(0), 86.13, 128.0, bpm_del_dsp=True)
        assert iv == pytest.approx(60.0 / 128.0, abs=1e-6)
        assert bpm == 128.0
        assert fb == 0.0

    def test_con_la_envolvente_encuentra_la_FASE(self):
        """Hasta el 2026-09-18 la rejilla del camino por trozos tenia un
        parametro `first_beat_offset` que NADIE le pasaba nunca, asi que
        `first_beat` salia 0.0 siempre: la rejilla arrancaba donde arranca el
        FICHERO. Lo que hay que comprobar es que la fase se BUSCA en el audio."""
        bpm, fase, fps = 128.0, 0.137, 86.13
        rnd = np.random.RandomState(1)
        n = int(360 * fps)
        onset = rnd.rand(n) * 0.05
        iv = 60.0 / bpm
        k = 0
        while fase + k * iv < 360:
            i = int(round((fase + k * iv) * fps))
            if 0 <= i < n:
                onset[i] += 2.2 if k % 4 == 0 else 1.0
            k += 1

        fb, iv, _ = rejilla_y_bpm(onset, fps, bpm, bpm_del_dsp=True)

        d = (fase - fb) % iv
        assert min(d, iv - d) < 0.012

    def test_sin_envolvente_no_se_inventa_la_fase(self):
        """Sin audio que mirar solo se puede decir el intervalo. Un 0.0 aqui es
        honesto; una fase inventada mueve la rejilla a un sitio que no es."""
        fb, iv, _ = rejilla_y_bpm(np.zeros(0), 86.13, 120.0, bpm_del_dsp=True)
        assert fb == 0.0
        assert iv == pytest.approx(0.5, abs=1e-6)

    def test_bpm_cero_no_divide_por_cero(self):
        """Un análisis fallido deja bpm=0; el grid tiene que degradar, no
        lanzar ZeroDivisionError."""
        _, iv, _ = rejilla_y_bpm(np.zeros(0), 86.13, 0.0, bpm_del_dsp=True)
        assert iv > 0


# ==================== ANÁLISIS DE UN CHUNK ====================

class TestAnalisisDeChunk:
    def test_energia_de_un_chunk_ordena(self, analyzer):
        flojo = analyzer.analyze_chunk_energy(_click(128, 8, amp=0.05), SR, 0.0)
        fuerte = analyzer.analyze_chunk_energy(_click(128, 8, amp=0.95), SR, 0.0)
        assert fuerte['energy_mean'] > flojo['energy_mean']
        assert fuerte['energy_max'] > flojo['energy_max']

    def test_la_curva_de_un_chunk_lleva_el_offset_temporal(self, analyzer):
        """Cada chunk aporta su tramo; si no desplaza el tiempo por su
        `chunk_start`, la curva combinada se apelmaza en el segundo 0."""
        r = analyzer.analyze_chunk_energy(_click(128, 8), SR, chunk_start=45.0)
        curva = r.get('energy_curve') or []
        assert curva, "el chunk no devolvió curva"
        assert min(p['time'] for p in curva) >= 45.0


# ==================== END TO END MULTI-CHUNK ====================

class TestFullAnalysis:
    def test_recorre_el_camino_multichunk_completo(self, analyzer):
        """60 s con chunks de 15 s = 4-5 chunks. Ejercita trocear, analizar
        cada chunk y fusionar, que es lo que corre en Render con cualquier
        track de más de 4 minutos."""
        y = np.concatenate([
            _click(128, 15, amp=0.15),   # intro floja
            _click(128, 30, amp=0.95),   # cuerpo
            _click(128, 15, amp=0.15),   # outro floja
        ])
        path = tempfile.mktemp(suffix=".wav")
        soundfile.write(path, y, SR)

        r = analyzer.full_analysis(path)

        assert abs(r['bpm'] - 128) / 128 < 0.08, f"bpm={r['bpm']}"
        assert r['duration'] == pytest.approx(60.0, abs=1.0)
        assert 1 <= r['energy_dj'] <= 10
        assert r['key'] is not None and r['camelot'] is not None
        assert r['beat_interval'] > 0
        assert r['has_intro'] or r['has_outro'], (
            "un track con entrada y salida suaves debería marcar intro u outro"
        )
        assert isinstance(r.get('cue_points'), list)
        assert isinstance(r.get('structure_sections'), list)

    def test_el_resultado_trae_lo_que_AnalysisResult_exige(self, analyzer):
        """El dict de full_analysis alimenta AnalysisResult; si falta un campo
        obligatorio, /analyze revienta con ValidationError en producción."""
        y = _click(128, 40, amp=0.7)
        path = tempfile.mktemp(suffix=".wav")
        soundfile.write(path, y, SR)

        r = analyzer.full_analysis(path)
        for campo in ('bpm', 'duration', 'energy_dj', 'key', 'camelot',
                      'track_type', 'first_beat', 'beat_interval'):
            assert campo in r, f"full_analysis no devuelve '{campo}'"
