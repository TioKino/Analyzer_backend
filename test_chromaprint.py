"""Tests para chromaprint_helper y migracion calculate_fingerprint.

Estos tests NO dependen de fpcalc real instalado: mockean
`acoustid.fingerprint_file` y `shutil.which` para validar la logica
de fallback y el formato de salida.

Tests con fpcalc real (ej: mismo audio en mp3 y flac da el mismo fp)
pertenecen a integracion y se ejecutan manualmente, no en CI.
"""

import hashlib
import os
import tempfile
from unittest.mock import patch

import pytest

import chromaprint_helper as cp


@pytest.fixture(autouse=True)
def reset_fpcalc_cache():
    """Resetea el cache de is_fpcalc_available entre tests."""
    cp._FPCALC_AVAILABLE = None
    yield
    cp._FPCALC_AVAILABLE = None


@pytest.fixture
def tmp_audio_file():
    """Crea un archivo temporal con bytes arbitrarios.

    Sirve para tests que solo necesitan que el path exista.
    NO es audio real — solo se usa para mockear acoustid.fingerprint_file.
    """
    fd, path = tempfile.mkstemp(suffix=".mp3")
    try:
        os.write(fd, b"FAKE_MP3_DATA" * 100)
        os.close(fd)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ==================== is_fpcalc_available ====================

class TestFpcalcDetection:
    def test_fpcalc_available_when_in_path(self):
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            assert cp.is_fpcalc_available() is True

    def test_fpcalc_not_available_when_missing(self):
        with patch("chromaprint_helper.shutil.which", return_value=None):
            assert cp.is_fpcalc_available() is False

    def test_fpcalc_check_is_cached(self):
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc") as mock_which:
            cp.is_fpcalc_available()
            cp.is_fpcalc_available()
            cp.is_fpcalc_available()
            assert mock_which.call_count == 1


# ==================== calculate_chromaprint_fingerprint ====================

class TestCalculateChromaprintFingerprint:
    def test_raises_unavailable_when_fpcalc_missing(self, tmp_audio_file):
        with patch("chromaprint_helper.shutil.which", return_value=None):
            with pytest.raises(cp.ChromaprintUnavailable):
                cp.calculate_chromaprint_fingerprint(tmp_audio_file)

    def test_returns_md5_hex_format(self, tmp_audio_file):
        """El fingerprint devuelto debe ser hex de 32 chars (MD5)."""
        fake_fp_str = "AQADtEmSJEqUNDeG4j2RyEsiMXCQpIeJ"
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            with patch("acoustid.fingerprint_file", return_value=(180.5, fake_fp_str)):
                fp_md5, fp_b64, dur_ms = cp.calculate_chromaprint_fingerprint(tmp_audio_file)

        assert isinstance(fp_md5, str)
        assert len(fp_md5) == 32
        assert all(c in "0123456789abcdef" for c in fp_md5)
        assert dur_ms == 180500

    def test_deterministic_for_same_input(self, tmp_audio_file):
        """Llamadas con el mismo fp_str dan el mismo fp_md5."""
        fake_fp_str = "AQADtEmSJEqUNDeG4j2RyEsiMXCQpIeJ"
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            with patch("acoustid.fingerprint_file", return_value=(180.0, fake_fp_str)):
                fp_a, _, _ = cp.calculate_chromaprint_fingerprint(tmp_audio_file)
                fp_b, _, _ = cp.calculate_chromaprint_fingerprint(tmp_audio_file)
        assert fp_a == fp_b

    def test_raises_failed_on_empty_fingerprint(self, tmp_audio_file):
        """fpcalc devolviendo cadena vacia (audio <30s) debe ser ChromaprintFailed."""
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            with patch("acoustid.fingerprint_file", return_value=(10.0, "")):
                with pytest.raises(cp.ChromaprintFailed):
                    cp.calculate_chromaprint_fingerprint(tmp_audio_file)

    def test_raises_failed_on_acoustid_error(self, tmp_audio_file):
        """Si acoustid lanza FingerprintGenerationError, propagamos como ChromaprintFailed."""
        import acoustid
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            with patch(
                "acoustid.fingerprint_file",
                side_effect=acoustid.FingerprintGenerationError("decode failed"),
            ):
                with pytest.raises(cp.ChromaprintFailed):
                    cp.calculate_chromaprint_fingerprint(tmp_audio_file)


# ==================== main.calculate_fingerprint (fallback path) ====================

# main.py importa fastapi/librosa al cargarse; si esas deps no estan en el entorno
# de tests, marcamos la clase entera como skip para no bloquear suites parciales.
def _main_importable() -> bool:
    try:
        import importlib.util
        for mod in ("fastapi", "librosa"):
            if importlib.util.find_spec(mod) is None:
                return False
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _main_importable(),
    reason="main.py requiere fastapi+librosa instaladas",
)
class TestCalculateFingerprintFallback:
    """Verifica que main.calculate_fingerprint cae a MD5 cuando Chromaprint no esta disponible."""

    def test_fallback_to_md5_when_fpcalc_missing(self, tmp_audio_file):
        from main import calculate_fingerprint

        with patch("chromaprint_helper.shutil.which", return_value=None):
            fp, source = calculate_fingerprint(tmp_audio_file)

        # MD5 del contenido del archivo.
        with open(tmp_audio_file, "rb") as f:
            expected_md5 = hashlib.md5(f.read()).hexdigest()

        assert fp == expected_md5
        assert source == "md5_legacy"
        assert len(fp) == 32

    def test_uses_chromaprint_when_available(self, tmp_audio_file):
        from main import calculate_fingerprint

        fake_fp_str = "AQADtEmSJEqUNDeG4j2RyEsiMXCQpIeJ"
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            with patch("acoustid.fingerprint_file", return_value=(180.0, fake_fp_str)):
                fp, source = calculate_fingerprint(tmp_audio_file)

        assert source == "chromaprint"
        assert len(fp) == 32

    def test_chromaprint_and_md5_give_different_fingerprints(self, tmp_audio_file):
        """Sanity: Chromaprint y MD5(content) NO devuelven el mismo valor.

        Si lo hicieran, no podriamos distinguir tracks legacy de migrados.
        """
        from main import calculate_fingerprint

        with patch("chromaprint_helper.shutil.which", return_value=None):
            fp_legacy, _ = calculate_fingerprint(tmp_audio_file)

        fake_fp_str = "AQADtEmSJEqUNDeG4j2RyEsiMXCQpIeJ"
        with patch("chromaprint_helper.shutil.which", return_value="/usr/bin/fpcalc"):
            with patch("acoustid.fingerprint_file", return_value=(180.0, fake_fp_str)):
                fp_chromaprint, _ = calculate_fingerprint(tmp_audio_file)

        assert fp_legacy != fp_chromaprint
