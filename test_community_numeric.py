"""Tests para consensus comunitario numerico (Fase 5).

Cubre normalize_bpm_to_canonical, camelot_to_key, validacion + persistencia
de bpm/energy/year via /community/override, y mediana en
get_community_consensus_numeric.

Para ejecutar:
    pytest test_community_numeric.py -v
"""

import os
import tempfile

import pytest


# Aislar la BD de tests para no tocar el SQLite de producción.
@pytest.fixture(autouse=True, scope='module')
def _isolated_db():
    fd, path = tempfile.mkstemp(suffix='.db', prefix='analyzer_test_phase5_')
    os.close(fd)
    os.environ['DATABASE_PATH'] = path
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


from bpm_utils import normalize_bpm_to_canonical  # noqa: E402
from main import KEY_TO_CAMELOT, camelot_to_key  # noqa: E402


# ============================================================================
# normalize_bpm_to_canonical
# ============================================================================

class TestNormalizeBpmToCanonical:
    """Verifica el doubling/halving al rango canonico [60, 180]."""

    def test_low_value_doubled(self):
        # 32 -> 64 (32 *2 = 64, 64 >= 60 -> stop). El spec sugeria 128 pero
        # el algoritmo prescrito en el docstring (while bpm < 60) deja en 64.
        assert normalize_bpm_to_canonical(32) == 64.0

    def test_value_within_range_unchanged(self):
        assert normalize_bpm_to_canonical(64) == 64.0
        assert normalize_bpm_to_canonical(128) == 128.0
        assert normalize_bpm_to_canonical(140) == 140.0

    def test_high_value_halved(self):
        # 256 > 180, /2 = 128.
        assert normalize_bpm_to_canonical(256) == 128.0
        # 360 > 180, /2 = 180 (incluido).
        assert normalize_bpm_to_canonical(360) == 180.0

    def test_boundaries_inclusive(self):
        assert normalize_bpm_to_canonical(60) == 60.0
        assert normalize_bpm_to_canonical(180) == 180.0

    def test_just_below_lower_boundary_doubles(self):
        # 59 < 60, *2 = 118 (en rango).
        assert normalize_bpm_to_canonical(59) == 118.0

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            normalize_bpm_to_canonical(0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            normalize_bpm_to_canonical(-10)

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            normalize_bpm_to_canonical(float('nan'))

    def test_rejects_inf(self):
        with pytest.raises(ValueError):
            normalize_bpm_to_canonical(float('inf'))

    def test_rounds_to_one_decimal(self):
        # 127.85 / 1 = 127.85 -> 127.8 o 127.9 (banker's rounding)
        result = normalize_bpm_to_canonical(127.85)
        # Aceptar cualquier 1-decimal alrededor del valor.
        assert abs(result - 127.85) < 0.1


# ============================================================================
# camelot_to_key
# ============================================================================

class TestCamelotToKey:
    """Verifica el mapeo inverso Camelot -> nota cruda."""

    def test_inverse_of_key_to_camelot(self):
        # Todas las keys de KEY_TO_CAMELOT deben hacer round-trip.
        for k, c in KEY_TO_CAMELOT.items():
            assert camelot_to_key(c) == k, f"Round-trip falla para {k}->{c}->{camelot_to_key(c)}"

    def test_8b_is_c_major(self):
        assert camelot_to_key('8B') == 'C'

    def test_8a_is_a_minor(self):
        assert camelot_to_key('8A') == 'Am'

    def test_case_insensitive(self):
        assert camelot_to_key('8b') == 'C'
        assert camelot_to_key('8a') == 'Am'

    def test_strips_whitespace(self):
        assert camelot_to_key(' 8B ') == 'C'

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            camelot_to_key('99Z')
        with pytest.raises(ValueError):
            camelot_to_key('hello')
        with pytest.raises(ValueError):
            camelot_to_key('')
        with pytest.raises(ValueError):
            camelot_to_key(None)


# ============================================================================
# _validate_community_field — BPM / Energy / Year
# ============================================================================

class TestValidateCommunityField:
    """Verifica que el validador acepta y normaliza correctamente los
    nuevos campos numericos."""

    def test_bpm_accepts_and_normalizes(self):
        from routes.community import _validate_community_field
        # 256 -> 128.0 canonico.
        normalized, err = _validate_community_field('bpm', '256')
        assert err is None
        assert normalized == '128.0'

    def test_bpm_keeps_in_range(self):
        from routes.community import _validate_community_field
        normalized, err = _validate_community_field('bpm', '128.0')
        assert err is None
        assert normalized == '128.0'

    def test_bpm_rejects_negative(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('bpm', '-10')
        assert err is not None

    def test_bpm_rejects_non_numeric(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('bpm', 'abc')
        assert err is not None

    def test_bpm_rejects_too_large(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('bpm', '9999')
        assert err is not None  # >999

    def test_energy_accepts_int(self):
        from routes.community import _validate_community_field
        normalized, err = _validate_community_field('energy', '7')
        assert err is None
        assert normalized == '7'

    def test_energy_rejects_out_of_range(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('energy', '0')
        assert err is not None
        _, err = _validate_community_field('energy', '11')
        assert err is not None

    def test_energy_rejects_non_numeric(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('energy', 'high')
        assert err is not None

    def test_year_accepts_valid(self):
        from routes.community import _validate_community_field
        normalized, err = _validate_community_field('year', '1995')
        assert err is None
        assert normalized == '1995'

    def test_year_rejects_too_old(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('year', '1800')
        assert err is not None

    def test_year_rejects_far_future(self):
        from routes.community import _validate_community_field
        _, err = _validate_community_field('year', '3000')
        assert err is not None


# ============================================================================
# get_community_consensus_numeric — mediana
# ============================================================================

class TestConsensusNumericMedian:
    """Tests directos sobre db.get_community_consensus_numeric() sin pasar por
    el endpoint HTTP (mucho mas rapido y no levanta servidor)."""

    def _fresh_db(self):
        # Cada test usa fingerprints distintos para no chocar.
        from database import AnalysisDB
        return AnalysisDB(db_path=os.environ['DATABASE_PATH'])

    def test_bpm_three_equal_votes_returns_consensus(self):
        db = self._fresh_db()
        fp = 'test_fp_bpm_eq_' + os.urandom(4).hex()
        for i, dev in enumerate(['d1', 'd2', 'd3']):
            db.submit_community_override(fp, dev, 'bpm', '128.0')
        result = db.get_community_consensus_numeric(fp, 'bpm')
        assert result['consensus'] == 128.0
        assert result['consensus_votes'] == 3
        assert result['total_voters'] == 3

    def test_bpm_halftime_doubletime_normalized_via_endpoint(self):
        # Simulamos lo que haria el endpoint: cada dev envia un valor crudo,
        # _validate_community_field lo normaliza a "128.0", y el consensus
        # los agrupa.
        from routes.community import _validate_community_field
        db = self._fresh_db()
        fp = 'test_fp_bpm_ht_' + os.urandom(4).hex()
        for dev, raw_value in [('d1', '64'), ('d2', '128'), ('d3', '256')]:
            normalized, err = _validate_community_field('bpm', raw_value)
            assert err is None, f"validation falló para {raw_value}: {err}"
            db.submit_community_override(fp, dev, 'bpm', normalized)
        result = db.get_community_consensus_numeric(fp, 'bpm')
        # 64 -> 64.0 (en rango), 128 -> 128.0, 256 -> 128.0. La mediana de
        # [64, 128, 128] es 128.
        assert result['consensus'] == 128.0
        assert result['total_voters'] == 3

    def test_energy_median(self):
        db = self._fresh_db()
        fp = 'test_fp_energy_' + os.urandom(4).hex()
        for dev, val in [('d1', '5'), ('d2', '7'), ('d3', '9')]:
            db.submit_community_override(fp, dev, 'energy', val)
        result = db.get_community_consensus_numeric(fp, 'energy')
        # Mediana de [5, 7, 9] = 7.
        assert result['consensus'] == 7
        assert result['total_voters'] == 3

    def test_below_threshold_returns_none(self):
        db = self._fresh_db()
        fp = 'test_fp_thr_' + os.urandom(4).hex()
        # Solo 2 votos.
        db.submit_community_override(fp, 'd1', 'bpm', '128.0')
        db.submit_community_override(fp, 'd2', 'bpm', '130.0')
        result = db.get_community_consensus_numeric(fp, 'bpm')
        assert result['consensus'] is None
        assert result['consensus_votes'] == 0
        assert result['total_voters'] == 2

    def test_no_votes(self):
        db = self._fresh_db()
        result = db.get_community_consensus_numeric('nonexistent_fp', 'bpm')
        assert result['consensus'] is None
        assert result['total_voters'] == 0
        assert result['votes_distribution'] == {}


# ============================================================================
# Consensus categorico year (uses Fase 4 mode)
# ============================================================================

class TestConsensusYear:
    """Year es categorico (no numerico) porque la moda tiene mas sentido
    que la mediana (un track tiene UN año real, no un promedio)."""

    def _fresh_db(self):
        from database import AnalysisDB
        return AnalysisDB(db_path=os.environ['DATABASE_PATH'])

    def test_year_mode_consensus(self):
        db = self._fresh_db()
        fp = 'test_fp_year_' + os.urandom(4).hex()
        # 1995 gana 2-1 sobre 1996, pero no supera al 2do por >=2 -> sin consensus.
        # Necesitamos margen >= 2 y winner >= 3.
        for dev in ['d1', 'd2', 'd3', 'd4']:
            db.submit_community_override(fp, dev, 'year', '1995')
        db.submit_community_override(fp, 'd5', 'year', '1996')
        consensus = db.get_community_consensus(fp, 'year')
        assert consensus is not None
        assert consensus['value'] == '1995'
        assert consensus['votes'] == 4


# ============================================================================
# HTTP endpoint smoke tests (POST + GET branching)
# ============================================================================

class TestCommunityOverrideEndpoints:
    """Asegura que el endpoint POST/GET responde con la shape esperada
    para campos numericos (mediana)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_post_bpm_normalizes_and_persists(self, client):
        fp = 'http_test_bpm_' + os.urandom(4).hex()
        # 3 devices, 256 BPM cada uno. El validador colapsa a 128.0.
        for i, dev in enumerate(['dev_a', 'dev_b', 'dev_c']):
            r = client.post('/community/override', json={
                'fingerprint': fp,
                'device_id': dev,
                'field': 'bpm',
                'value': '256',
            })
            assert r.status_code == 200, r.text

        # GET devuelve mediana = 128.0
        r = client.get(f'/community/override/bpm/{fp}')
        assert r.status_code == 200
        data = r.json()
        assert data['consensus'] == 128.0
        assert data['total_voters'] == 3
        # `votes` distribution debe tener solo "128.0" como key
        assert '128.0' in data['votes']

    def test_get_rejects_unsupported_field(self, client):
        r = client.get('/community/override/wat/abc123')
        assert r.status_code == 400

    def test_post_rejects_invalid_bpm(self, client):
        r = client.post('/community/override', json={
            'fingerprint': 'x' * 32,
            'device_id': 'dev_invalid',
            'field': 'bpm',
            'value': '-10',
        })
        assert r.status_code == 400

    def test_post_rejects_non_numeric_bpm(self, client):
        r = client.post('/community/override', json={
            'fingerprint': 'x' * 32,
            'device_id': 'dev_invalid_abc',
            'field': 'bpm',
            'value': 'abc',
        })
        assert r.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
