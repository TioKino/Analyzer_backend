"""
Tests para el módulo de validación
"""

import pytest
from validation import (
    validate_bpm_range,
    validate_energy_range,
    validate_limit,
    sanitize_string,
    validate_key,
    validate_camelot,
    validate_track_type,
    validate_genre,
    validate_track_id,
    sanitize_filename,
    ValidationError,
    SimpleRateLimiter,
    MIN_BPM, MAX_BPM,
    MIN_ENERGY, MAX_ENERGY,
    MAX_SEARCH_LIMIT,
)


# ============================================================================
# TESTS DE VALIDACIÓN DE RANGOS
# ============================================================================

class TestValidateBpmRange:
    """Tests para validate_bpm_range"""
    
    def test_valid_range(self):
        """Test rango válido"""
        min_bpm, max_bpm = validate_bpm_range(120, 130)
        assert min_bpm == 120
        assert max_bpm == 130
    
    def test_none_values_use_defaults(self):
        """Test que None usa valores por defecto"""
        min_bpm, max_bpm = validate_bpm_range(None, None)
        assert min_bpm >= MIN_BPM
        assert max_bpm <= MAX_BPM
    
    def test_clamps_to_min(self):
        """Test que clampea al mínimo"""
        min_bpm, max_bpm = validate_bpm_range(30, 130)
        assert min_bpm == MIN_BPM
    
    def test_clamps_to_max(self):
        """Test que clampea al máximo"""
        min_bpm, max_bpm = validate_bpm_range(100, 250)
        assert max_bpm == MAX_BPM
    
    def test_swaps_if_inverted(self):
        """Test que intercambia si está invertido"""
        min_bpm, max_bpm = validate_bpm_range(150, 100)
        assert min_bpm == 100
        assert max_bpm == 150


class TestValidateEnergyRange:
    """Tests para validate_energy_range"""
    
    def test_valid_range(self):
        """Test rango válido"""
        min_e, max_e = validate_energy_range(3, 8)
        assert min_e == 3
        assert max_e == 8
    
    def test_clamps_to_limits(self):
        """Test que clampea a límites"""
        min_e, max_e = validate_energy_range(-5, 20)
        assert min_e == MIN_ENERGY
        assert max_e == MAX_ENERGY
    
    def test_swaps_if_inverted(self):
        """Test que intercambia si está invertido"""
        min_e, max_e = validate_energy_range(8, 3)
        assert min_e == 3
        assert max_e == 8


class TestValidateLimit:
    """Tests para validate_limit"""
    
    def test_valid_limit(self):
        """Test límite válido"""
        assert validate_limit(50) == 50
    
    def test_zero_returns_one(self):
        """Test que 0 retorna 1"""
        assert validate_limit(0) == 1
    
    def test_negative_returns_one(self):
        """Test que negativo retorna 1"""
        assert validate_limit(-10) == 1
    
    def test_exceeds_max_returns_max(self):
        """Test que exceder máximo retorna máximo"""
        assert validate_limit(5000) == MAX_SEARCH_LIMIT
    
    def test_custom_max(self):
        """Test límite máximo personalizado"""
        assert validate_limit(100, max_limit=50) == 50


# ============================================================================
# TESTS DE SANITIZACIÓN DE STRINGS
# ============================================================================

class TestSanitizeString:
    """Tests para sanitize_string"""
    
    def test_normal_string(self):
        """Test string normal"""
        assert sanitize_string("Hello World") == "Hello World"
    
    def test_removes_extra_spaces(self):
        """Test que elimina espacios extra"""
        assert sanitize_string("  Hello   World  ") == "Hello World"
    
    def test_truncates_long_string(self):
        """Test que trunca strings largos"""
        long_str = "a" * 1000
        result = sanitize_string(long_str, max_length=100)
        assert len(result) == 100
    
    def test_removes_dangerous_chars(self):
        """Test que elimina caracteres peligrosos"""
        result = sanitize_string("Hello<script>alert()</script>")
        assert "<" not in result
        assert ">" not in result
    
    def test_detects_sql_injection(self):
        """Test que detecta SQL injection"""
        with pytest.raises(ValidationError):
            sanitize_string("'; DROP TABLE users; --")
    
    def test_none_returns_empty(self):
        """Test que None retorna vacío si permitido"""
        assert sanitize_string(None, allow_empty=True) == ""
    
    def test_none_raises_if_required(self):
        """Test que None lanza error si requerido"""
        with pytest.raises(ValidationError):
            sanitize_string(None, allow_empty=False)


class TestSanitizeFilename:
    """Tests para sanitize_filename"""
    
    def test_normal_filename(self):
        """Test nombre normal"""
        assert sanitize_filename("track.mp3") == "track.mp3"
    
    def test_removes_path_traversal(self):
        """Test que elimina path traversal"""
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
    
    def test_removes_dangerous_chars(self):
        """Test que elimina caracteres peligrosos"""
        result = sanitize_filename('track<>:"|?*.mp3')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
    
    def test_truncates_long_filename(self):
        """Test que trunca nombres largos"""
        long_name = "a" * 300 + ".mp3"
        result = sanitize_filename(long_name)
        assert len(result) <= 255


# ============================================================================
# TESTS DE VALIDACIÓN DE KEYS
# ============================================================================

class TestValidateKey:
    """Tests para validate_key"""
    
    def test_valid_major_key(self):
        """Test tonalidad mayor válida"""
        assert validate_key("C") == "C"
        assert validate_key("G") == "G"
    
    def test_valid_minor_key(self):
        """Test tonalidad menor válida"""
        assert validate_key("Am") == "Am"
        assert validate_key("F#m") == "F#m"
    
    def test_normalizes_format(self):
        """Test que normaliza formato"""
        assert validate_key("A minor") == "Am"
        assert validate_key("C Major") == "C"
    
    def test_invalid_key_raises(self):
        """Test que key inválida lanza error"""
        with pytest.raises(ValidationError):
            validate_key("X#")
    
    def test_empty_key_raises(self):
        """Test que key vacía lanza error"""
        with pytest.raises(ValidationError):
            validate_key("")


class TestValidateCamelot:
    """Tests para validate_camelot"""
    
    def test_valid_camelot(self):
        """Test Camelot válido"""
        assert validate_camelot("8A") == "8A"
        assert validate_camelot("11B") == "11B"
    
    def test_normalizes_case(self):
        """Test que normaliza mayúsculas"""
        assert validate_camelot("8a") == "8A"
        assert validate_camelot("11b") == "11B"
    
    def test_invalid_camelot_raises(self):
        """Test que Camelot inválido lanza error"""
        with pytest.raises(ValidationError):
            validate_camelot("13A")
        with pytest.raises(ValidationError):
            validate_camelot("0B")


class TestValidateTrackType:
    """Tests para validate_track_type"""
    
    def test_valid_types(self):
        """Test tipos válidos"""
        assert validate_track_type("warmup") == "warmup"
        assert validate_track_type("peak") == "peak"
        assert validate_track_type("closing") == "closing"
    
    def test_normalizes_case(self):
        """Test que normaliza mayúsculas"""
        assert validate_track_type("PEAK") == "peak"
        assert validate_track_type("WarmUp") == "warmup"
    
    def test_empty_returns_all(self):
        """Test que vacío retorna 'all'"""
        assert validate_track_type("") == "all"
        assert validate_track_type(None) == "all"
    
    def test_invalid_type_raises(self):
        """Test que tipo inválido lanza error"""
        with pytest.raises(ValidationError):
            validate_track_type("invalid")


class TestValidateGenre:
    """Tests para validate_genre"""
    
    def test_capitalizes_genre(self):
        """Test que capitaliza género"""
        assert validate_genre("tech house") == "Tech House"
        assert validate_genre("TECHNO") == "Techno"
    
    def test_sanitizes_dangerous_chars(self):
        """Test que sanitiza caracteres peligrosos"""
        result = validate_genre("House<script>")
        assert "<" not in result


class TestValidateTrackId:
    """Tests para validate_track_id"""
    
    def test_valid_md5(self):
        """Test MD5 válido"""
        valid_md5 = "d41d8cd98f00b204e9800998ecf8427e"
        assert validate_track_id(valid_md5) == valid_md5
    
    def test_normalizes_case(self):
        """Test que normaliza a minúsculas"""
        upper_md5 = "D41D8CD98F00B204E9800998ECF8427E"
        assert validate_track_id(upper_md5) == upper_md5.lower()
    
    def test_empty_raises(self):
        """Test que vacío lanza error"""
        with pytest.raises(ValidationError):
            validate_track_id("")
    
    def test_invalid_format_raises(self):
        """Test que formato inválido lanza error"""
        with pytest.raises(ValidationError):
            validate_track_id("not-a-valid-id!")


# ============================================================================
# TESTS DE RATE LIMITER
# ============================================================================

class TestSimpleRateLimiter:
    """Tests para SimpleRateLimiter"""
    
    def test_allows_first_request(self):
        """Test que permite primera solicitud"""
        limiter = SimpleRateLimiter(requests_per_minute=10)
        assert limiter.is_allowed("client1") is True
    
    def test_allows_multiple_within_limit(self):
        """Test que permite múltiples dentro del límite"""
        limiter = SimpleRateLimiter(requests_per_minute=10)
        for _ in range(5):
            assert limiter.is_allowed("client2") is True
    
    def test_blocks_when_exceeded(self):
        """Test que bloquea cuando se excede"""
        limiter = SimpleRateLimiter(requests_per_minute=3)
        for _ in range(3):
            limiter.is_allowed("client3")
        assert limiter.is_allowed("client3") is False
    
    def test_different_clients_independent(self):
        """Test que clientes diferentes son independientes"""
        limiter = SimpleRateLimiter(requests_per_minute=2)
        limiter.is_allowed("clientA")
        limiter.is_allowed("clientA")
        # clientA está al límite
        assert limiter.is_allowed("clientA") is False
        # clientB es independiente
        assert limiter.is_allowed("clientB") is True
    
    def test_get_remaining(self):
        """Test obtener requests restantes"""
        limiter = SimpleRateLimiter(requests_per_minute=10)
        limiter.is_allowed("client4")
        limiter.is_allowed("client4")
        assert limiter.get_remaining("client4") == 8


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
