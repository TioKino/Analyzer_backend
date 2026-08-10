"""
ChunkedAudioAnalyzer — la ruta de los tracks > 4 min.

POR QUÉ IMPORTA: en Render, todo track de más de 4 minutos va por aquí
(`main.CHUNK_ANALYSIS_THRESHOLD`), o sea buena parte de lo que analiza un DJ.
El módulo estaba al 15 % de cobertura y lo único probado era
`fuse_bpm_results` (en `test_bpm_fusion.py`). Toda la fusión de tonalidad, la
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


# ==================== FUSIÓN DE TONALIDAD ====================

class TestFusionDeTonalidad:
    def test_gana_la_key_con_mas_peso(self, analyzer):
        chunks = [
            {'key': 'Am', 'confidence': 0.9},
            {'key': 'Am', 'confidence': 0.8},
            {'key': 'C',  'confidence': 0.3},
        ]
        r = analyzer.fuse_key_results(chunks)
        assert r['key'] == 'Am'
        assert r['scale'] == 'minor'
        assert r['camelot'] == '8A'
        assert 0.0 <= r['confidence'] <= 1.0

    def test_los_chunks_con_mas_energia_pesan_mas(self, analyzer):
        """El drop define la tonalidad mejor que una intro de pads: si el
        chunk con energía manda, la fusión debe seguirle."""
        chunks = [
            {'key': 'C',  'confidence': 0.6},   # intro floja
            {'key': 'F#m', 'confidence': 0.6},  # drop
        ]
        r = analyzer.fuse_key_results(chunks, energy_weights=[0.05, 0.95])
        assert r['key'] == 'F#m', "la fusión ignoró los pesos de energía"

    def test_key_None_no_revienta(self, analyzer):
        """REGRESIÓN (panel admin 2026-05-20): chunks degenerados llegaban con
        `'key': None` literal. `r.get('key', 'C')` no aplica el default cuando
        el valor existe y es None, así que best_key acababa en None y
        `best_key.endswith('m')` petaba con AttributeError."""
        r = analyzer.fuse_key_results([
            {'key': None, 'confidence': 0.5},
            {'key': None, 'confidence': 0.5},
        ])
        assert r['key'] == 'C'
        assert r['scale'] == 'major'

    def test_confianza_None_tampoco_revienta(self, analyzer):
        """REGRESIÓN ENCONTRADA POR ESTE TEST (2026-08-09): el mismo fallo del
        default de dict que se había corregido para 'key' seguía vivo dos
        líneas más arriba, en la lista de pesos. Con `'confidence': None`,
        `sum(energy_weights)` lanzaba TypeError y tumbaba el análisis del track
        completo, no solo el chunk."""
        r = analyzer.fuse_key_results([{'key': 'Am', 'confidence': None}])
        assert r['key'] == 'Am'

    def test_sin_chunks_devuelve_un_default_utilizable(self, analyzer):
        r = analyzer.fuse_key_results([])
        assert r['key'] == 'C' and r['camelot'] == '8B'
        assert r['confidence'] == 0.0

    def test_el_camelot_nunca_contradice_a_la_key(self, analyzer):
        from chunked_analyzer import KEY_TO_CAMELOT

        for key in ('C', 'Am', 'F#m', 'G', 'D#m'):
            r = analyzer.fuse_key_results([{'key': key, 'confidence': 0.9}])
            assert r['camelot'] == KEY_TO_CAMELOT[key], f"{key} -> {r['camelot']}"


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
        assert analyzer._calculate_energy_dj(raw) == esperado

    def test_siempre_dentro_de_1_10(self, analyzer):
        for raw in (0.0, 0.05, 0.1, 0.2, 0.3, 0.41, 0.5, 5.0):
            v = analyzer._calculate_energy_dj(raw)
            assert 1 <= v <= 10, f"raw={raw} -> {v}"

    def test_es_monotona(self, analyzer):
        vals = [analyzer._calculate_energy_dj(r) for r in
                (0.03, 0.08, 0.15, 0.25, 0.35, 0.41)]
        assert vals == sorted(vals), f"la escala no es monótona: {vals}"

    @pytest.mark.parametrize("malo", [float('nan'), float('inf'), float('-inf')])
    def test_nan_e_inf_no_revientan(self, analyzer, malo):
        """REGRESIÓN: era el error #1 del panel admin (112 ocurrencias). Un RMS
        NaN hacía que las comparaciones dieran False y `int(NaN)` explotara con
        ValueError, tumbando el análisis entero."""
        v = analyzer._calculate_energy_dj(malo)
        assert 1 <= v <= 10 and not math.isnan(v)


# ==================== BEAT GRID ====================

class TestBeatGrid:
    def test_el_intervalo_es_60_partido_bpm(self, analyzer):
        g = analyzer.calculate_beat_grid(128.0)
        assert g['beat_interval'] == pytest.approx(60.0 / 128.0, abs=1e-6)
        assert g['bpm'] == 128.0
        assert g['first_beat'] == 0.0

    def test_respeta_el_offset_del_primer_beat(self, analyzer):
        g = analyzer.calculate_beat_grid(120.0, first_beat_offset=0.35)
        assert g['first_beat'] == 0.35

    def test_bpm_cero_no_divide_por_cero(self, analyzer):
        """Un análisis fallido deja bpm=0; el grid tiene que degradar, no
        lanzar ZeroDivisionError."""
        g = analyzer.calculate_beat_grid(0.0)
        assert g['beat_interval'] > 0


# ==================== ANÁLISIS DE UN CHUNK ====================

class TestAnalisisDeChunk:
    def test_bpm_de_un_chunk_con_pulso_conocido(self, analyzer):
        r = analyzer.analyze_chunk_bpm(_click(128, 12), SR)
        assert abs(r['bpm'] - 128) / 128 < 0.08, f"bpm={r['bpm']}"

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
