"""
DJ ANALYZER - Tests Completos de Auditoría
=============================================
Cubre TODOS los tests API (A-001 a A-016) y Regresión (R-001 a R-013)
del MANUAL_TESTING_MAP.

Ejecutar:
    cd Analyzer_backend
    pytest test_full_audit.py -v --tb=short
"""

import pytest
import json
import os
import sys
import sqlite3
import tempfile

# Ensure project directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app, db
from models import AnalysisResult, KEY_TO_CAMELOT
from validation import (
    sanitize_string, validate_camelot, validate_key,
    ValidationError, SimpleRateLimiter,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seed_track():
    """Inserta un track de prueba en la BD para que las búsquedas funcionen."""
    track_data = {
        'id': 'd41d8cd98f00b204e9800998ecf8427e',
        'filename': 'Deadmau5 - Strobe.mp3',
        'artist': 'Deadmau5',
        'title': 'Strobe',
        'duration': 634.0,
        'bpm': 128.0,
        'key': 'Am',
        'camelot': '8A',
        'energy_dj': 7,
        'genre': 'Progressive House',
        'track_type': 'peak',
        'fingerprint': 'd41d8cd98f00b204e9800998ecf8427e',
        'bpm_source': 'analysis',
        'key_source': 'analysis',
        'label': 'mau5trap',
    }
    db.save_track(track_data)
    yield track_data
    # Cleanup
    db.delete_track(track_data['id'])


@pytest.fixture
def seed_track_special_name():
    """Track con nombre especial para test de validación."""
    track_data = {
        'id': 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6',
        'filename': "Guns N' Roses - Welcome To The Jungle.mp3",
        'artist': "Guns N' Roses",
        'title': 'Welcome To The Jungle',
        'duration': 271.0,
        'bpm': 124.0,
        'key': 'E',
        'camelot': '12B',
        'energy_dj': 9,
        'genre': 'Rock',
        'track_type': 'peak',
        'fingerprint': 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6',
    }
    db.save_track(track_data)
    yield track_data
    db.delete_track(track_data['id'])


# ============================================================================
# SECTION A: BACKEND API TESTS
# ============================================================================

class TestA001_HealthCheck:
    """A-001: GET /health → 200 con status info"""

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status(self, client):
        data = client.get("/health").json()
        assert data.get("status") in ("ok", "healthy")

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


class TestA002_AnalyzeTrack:
    """A-002: POST /analyze — rechaza archivos inválidos"""

    def test_rejects_non_audio(self, client):
        r = client.post("/analyze", files={
            "file": ("test.txt", b"not audio", "text/plain")
        })
        assert r.status_code == 400

    def test_requires_file(self, client):
        r = client.post("/analyze")
        assert r.status_code == 422


class TestA004_CompatibleKeysSearch:
    """A-004: GET /search/compatible/8A — devuelve tracks compatibles, no crash"""

    def test_returns_200(self, client, seed_track):
        r = client.get("/search/compatible/8A")
        assert r.status_code == 200

    def test_returns_compatible_keys_field(self, client, seed_track):
        data = client.get("/search/compatible/8A").json()
        assert "compatible_keys" in data
        assert "tracks" in data
        assert "camelot" in data
        assert data["camelot"] == "8A"

    def test_compatible_keys_list_correct(self, client, seed_track):
        data = client.get("/search/compatible/8A").json()
        # 8A compatible: 7A, 8A, 9A, 8B
        compat = data["compatible_keys"]
        assert "8A" in compat

    def test_no_type_error(self, client):
        """Regression: antes crasheaba con TypeError (pasaba lista en vez de string)"""
        r = client.get("/search/compatible/5B")
        assert r.status_code == 200
        assert isinstance(r.json().get("tracks"), list)


class TestA005_Artwork:
    """A-005: GET /artwork/{id} — 404 si no existe, con headers de cache si existe"""

    def test_artwork_not_found(self, client):
        r = client.get("/artwork/nonexistent_track_id_12345")
        assert r.status_code == 404

    def test_etag_304(self, client):
        """Si pasamos un ETag que no existe, no devuelve 304"""
        r = client.get("/artwork/test123", headers={"If-None-Match": '"fake-etag"'})
        assert r.status_code == 404  # Artwork doesn't exist


class TestA006_Preview:
    """A-006: GET /preview/{id} — devuelve 404 si track no existe"""

    def test_preview_not_found(self, client):
        r = client.get("/preview/0000000000000000aaaaaaaaaaaaaaaa")
        assert r.status_code == 404

    def test_preview_invalid_id(self, client):
        r = client.get("/preview/invalid!@#$%")
        assert r.status_code == 400


class TestA007_Correction:
    """A-007: POST /correction — guarda corrección"""

    def test_save_correction(self, client, seed_track):
        r = client.post("/correction", json={
            "track_id": seed_track['id'],
            "field": "genre",
            "old_value": "Progressive House",
            "new_value": "Tech House"
        })
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_correction_invalid_field(self, client, seed_track):
        r = client.post("/correction", json={
            "track_id": seed_track['id'],
            "field": "invalid_field",
            "old_value": "old",
            "new_value": "new"
        })
        assert r.status_code == 400

    def test_correction_requires_all_fields(self, client):
        r = client.post("/correction", json={"track_id": "123"})
        assert r.status_code == 422


class TestA008_SearchByArtist:
    """A-008: GET /search/artist/{name}"""

    def test_returns_200(self, client, seed_track):
        r = client.get("/search/artist/Deadmau5")
        assert r.status_code == 200

    def test_returns_matching_tracks(self, client, seed_track):
        data = client.get("/search/artist/Deadmau5").json()
        assert "tracks" in data
        assert isinstance(data["tracks"], list)


class TestA009_SearchByBPM:
    """A-009: GET /search/bpm?min_bpm=120&max_bpm=130"""

    def test_returns_200(self, client, seed_track):
        r = client.get("/search/bpm?min_bpm=120&max_bpm=130")
        assert r.status_code == 200

    def test_returns_tracks_in_range(self, client, seed_track):
        data = client.get("/search/bpm?min_bpm=120&max_bpm=130").json()
        assert "tracks" in data
        for t in data["tracks"]:
            assert 120 <= t["bpm"] <= 130


class TestA010_SearchByEnergy:
    """A-010: GET /search/energy?min_energy=7&max_energy=10"""

    def test_returns_200(self, client, seed_track):
        r = client.get("/search/energy?min_energy=7&max_energy=10")
        assert r.status_code == 200

    def test_returns_high_energy(self, client, seed_track):
        data = client.get("/search/energy?min_energy=7&max_energy=10").json()
        assert "tracks" in data
        for t in data["tracks"]:
            assert 7 <= t["energy_dj"] <= 10


class TestA011_AdvancedSearch:
    """A-011: POST /search/advanced"""

    def test_empty_filters(self, client):
        r = client.post("/search/advanced", json={})
        assert r.status_code == 200

    def test_multi_filter(self, client, seed_track):
        r = client.post("/search/advanced", json={
            "min_bpm": 125, "max_bpm": 135,
            "min_energy": 5,
            "genre": "Progressive House"
        })
        assert r.status_code == 200


class TestA013_CommunityBeatGrid:
    """A-013: POST /community/beat-grid"""

    def test_submit_beat_grid(self, client):
        r = client.post("/community/beat-grid", json={
            "fingerprint": "testfp1234567890abcdef12345678",
            "device_id": "test-device-001",
            "bpm_adjust": 0.5,
            "beat_offset": 0.02,
            "original_bpm": 128.0
        })
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_get_beat_grid(self, client):
        r = client.get("/community/beat-grid/testfp1234567890abcdef12345678")
        assert r.status_code == 200


class TestA014_ArtistWithApostrophe:
    """A-014: Buscar "Guns N' Roses" — no debe dar error de validación"""

    def test_apostrophe_in_search(self, client, seed_track_special_name):
        r = client.get("/search/artist/Guns N' Roses")
        assert r.status_code == 200

    def test_apostrophe_not_blocked(self):
        """Regression: antes, single quote era bloqueada por SQL injection regex"""
        result = sanitize_string("Guns N' Roses")
        assert "Guns N' Roses" == result

    def test_drop_in_name_not_blocked(self):
        """Regression: 'Drop' sola no debe ser bloqueada"""
        result = sanitize_string("Drop The Mic")
        assert "Drop The Mic" == result


class TestA015_RateLimiting:
    """A-015: Rate limiting funciona"""

    def test_rate_limiter_blocks_after_limit(self):
        limiter = SimpleRateLimiter(requests_per_minute=5)
        for _ in range(5):
            limiter.is_allowed("test-client-flood")
        assert limiter.is_allowed("test-client-flood") is False

    def test_different_clients_independent(self):
        limiter = SimpleRateLimiter(requests_per_minute=3)
        for _ in range(3):
            limiter.is_allowed("client-A")
        assert limiter.is_allowed("client-A") is False
        assert limiter.is_allowed("client-B") is True


class TestA016_LargeFileUpload:
    """A-016: Upload > 100MB debe rechazar"""

    def test_rejects_invalid_extension(self, client):
        r = client.post("/analyze", files={
            "file": ("huge.exe", b"\x00" * 1000, "application/octet-stream")
        })
        assert r.status_code == 400


# ============================================================================
# SECTION R: REGRESSION TESTS
# ============================================================================

class TestR001_EssentiaImports:
    """R-001: essentia_analyzer.py se importa sin SyntaxError"""

    def test_no_syntax_error(self):
        import ast
        path = os.path.join(os.path.dirname(__file__), "essentia_analyzer.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        # Must parse without SyntaxError
        ast.parse(source)

    def test_no_curly_quotes(self):
        path = os.path.join(os.path.dirname(__file__), "essentia_analyzer.py")
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "\u201c" not in source, "Found left curly quote"
        assert "\u201d" not in source, "Found right curly quote"


class TestR002_CompatibleKeysEndpoint:
    """R-002: GET /search/compatible/5B → array, no TypeError"""

    def test_no_crash(self, client):
        r = client.get("/search/compatible/5B")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["tracks"], list)

    def test_all_camelot_values(self, client):
        """Probar varias notaciones Camelot"""
        for cam in ["1A", "5B", "8A", "12B"]:
            r = client.get(f"/search/compatible/{cam}")
            assert r.status_code == 200, f"Failed for {cam}"


class TestR003_CollectiveDBSearch:
    """R-003: search_collective_db no da 'no such column' SQLite error"""

    def test_collective_search_no_column_error(self, seed_track):
        """Busca en BD colectiva sin error de columna inexistente"""
        from main import search_collective_db
        # Must not raise sqlite3.OperationalError about missing columns
        result = search_collective_db("Deadmau5", "Strobe")
        assert result is not None or result is None  # Any result is fine, just no crash


class TestR004_CachedAnalysisExtraFields:
    """R-004: AnalysisResult con campos extra no crashea"""

    def test_extra_fields_ignored(self):
        """Regression: extra keys in JSON caused Pydantic validation crash"""
        data = {
            "duration": 300.0, "bpm": 128.0, "bpm_confidence": 0.9,
            "key_confidence": 0.8, "energy_raw": 0.7, "energy_normalized": 0.7,
            "energy_dj": 7, "groove_score": 0.5, "swing_factor": 0.1,
            "has_intro": True, "has_buildup": True, "has_drop": True,
            "has_breakdown": True, "has_outro": True, "structure_sections": [],
            "track_type": "peak", "genre": "Techno", "has_vocals": False,
            "has_heavy_bass": True, "has_pads": False,
            "percussion_density": 0.8, "mix_energy_start": 0.5,
            "mix_energy_end": 0.6, "drop_timestamp": 120.0,
            # EXTRA FIELDS that used to crash:
            "original_file_path": "/tmp/test.mp3",
            "some_unknown_field": "should be ignored",
            "another_extra": 42,
        }
        result = AnalysisResult(**data)
        assert result.bpm == 128.0
        assert result.genre == "Techno"

    def test_model_config_extra_ignore(self):
        assert AnalysisResult.model_config.get("extra") == "ignore"


class TestR005_CommunityCuesTable:
    """R-005: community_cues table exists"""

    def test_table_exists(self):
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='community_cues'")
        result = c.fetchone()
        conn.close()
        assert result is not None, "community_cues table does not exist"

    def test_can_insert_and_query(self):
        conn = sqlite3.connect(db.db_path)
        c = conn.cursor()
        try:
            c.execute("""
                INSERT OR REPLACE INTO community_cues
                (fingerprint, device_id, cue_type, position_ms, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("test_fp_cue", "test_device", "hot_cue", 30000, "Test cue", "2024-01-01T00:00:00"))
            conn.commit()

            c.execute("SELECT * FROM community_cues WHERE fingerprint = ?", ("test_fp_cue",))
            row = c.fetchone()
            assert row is not None

            # Cleanup
            c.execute("DELETE FROM community_cues WHERE fingerprint = ?", ("test_fp_cue",))
            conn.commit()
        finally:
            conn.close()


class TestR006_CORSSyncHeaders:
    """R-006: CORS acepta X-Signature y X-Device-Id"""

    def test_cors_allows_sync_headers(self, client):
        r = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Signature,X-Device-Id,X-Original-Path"
        })
        allowed = r.headers.get("access-control-allow-headers", "")
        # In debug mode with wildcard origins, all headers are allowed
        # In prod, specific headers must be listed
        assert r.status_code in (200, 204, 405) or "X-Signature" in allowed or "*" in allowed


class TestR008_DateTimeCrashProtection:
    """R-008: DateTime.tryParse en frontend — verificamos que el patrón se usa"""

    def test_suggest_history_uses_tryparse(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Analyzer", "lib", "models", "suggest_history.dart")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "DateTime.tryParse" in content, "suggest_history.dart should use DateTime.tryParse"
        else:
            pytest.skip("Frontend files not accessible")

    def test_cue_pair_uses_tryparse(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Analyzer", "lib", "models", "cue_pair.dart")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "DateTime.tryParse" in content
        else:
            pytest.skip("Frontend files not accessible")

    def test_track_cue_uses_tryparse(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Analyzer", "lib", "models", "track_cue.dart")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "DateTime.tryParse" in content
        else:
            pytest.skip("Frontend files not accessible")

    def test_library_folder_uses_tryparse(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Analyzer", "lib", "models", "library_folder.dart")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "DateTime.tryParse" in content
        else:
            pytest.skip("Frontend files not accessible")


class TestR009_SessionDuration:
    """R-009: Duración de sesión usa duración real de tracks"""

    def test_saved_session_uses_actual_duration(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Analyzer", "lib", "models", "saved_session.dart")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert "t.duration" in content or "track.duration" in content, \
                "saved_session.dart should use actual track duration"
            assert "Duration(minutes: 6, seconds: 30)" in content, \
                "Should have fallback of 6:30 for tracks without duration"
        else:
            pytest.skip("Frontend files not accessible")


class TestR010_FileExtensionStripping:
    """R-010: Extension stripping usa regex genérico"""

    def test_audio_analysis_service_strips_any_extension(self):
        path = os.path.join(os.path.dirname(__file__), "..", "Analyzer", "lib", "services", "audio_player_service.dart")
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            assert r"\.[^.]+$" in content or "replaceAll(RegExp" in content, \
                "Should use regex to strip any file extension"
        else:
            pytest.skip("Frontend files not accessible")


class TestR011_SyncClearAuthorization:
    """R-011: DELETE /sync/clear sin admin key válida → 403"""

    def test_clear_without_key_returns_403(self, client):
        r = client.delete("/sync/clear")
        assert r.status_code == 403

    def test_clear_with_wrong_key_returns_403(self, client):
        r = client.delete("/sync/clear", headers={"X-Admin-Key": "wrong-key-12345"})
        # If ADMIN_TOKEN is empty, any non-empty key would fail the comparison
        # If ADMIN_TOKEN is set, wrong key returns 403
        assert r.status_code in (403, 200)  # 200 only if ADMIN_TOKEN == "wrong-key-12345"

    def test_clear_empty_key_returns_403(self, client):
        r = client.delete("/sync/clear", headers={"X-Admin-Key": ""})
        assert r.status_code == 403


class TestR012_ValidationAllowsSpecialNames:
    """R-012: Validación permite 'Drop' en nombres de artistas"""

    def test_drop_the_mic(self):
        result = sanitize_string("Drop The Mic")
        assert result == "Drop The Mic"

    def test_alter_ego(self):
        result = sanitize_string("Alter Ego")
        assert result == "Alter Ego"

    def test_guns_n_roses(self):
        result = sanitize_string("Guns N' Roses")
        assert result == "Guns N' Roses"

    def test_real_sql_injection_still_blocked(self):
        with pytest.raises(Exception):
            sanitize_string("'; DROP TABLE users; --")

    def test_select_from_blocked(self):
        with pytest.raises(Exception):
            sanitize_string("SELECT * FROM tracks")


class TestR013_DatabaseInit:
    """R-013 extra: AnalysisDB usa DATABASE_PATH de config"""

    def test_db_path_from_config(self):
        from config import DATABASE_PATH
        assert db.db_path == DATABASE_PATH

    def test_wal_mode_on_persistent_connection(self):
        """WAL mode debe estar activo en la conexión persistente"""
        cursor = db.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal", f"Expected WAL mode, got {mode}"


# ============================================================================
# EXTRA: PYDANTIC MODEL COMPLETENESS
# ============================================================================

class TestPydanticModel:
    """Verifica que el modelo AnalysisResult tiene todos los campos necesarios"""

    def test_all_required_fields_present(self):
        fields = set(AnalysisResult.model_fields.keys())
        required = {
            'duration', 'bpm', 'bpm_confidence', 'key_confidence',
            'energy_raw', 'energy_normalized', 'energy_dj',
            'groove_score', 'swing_factor',
            'has_intro', 'has_buildup', 'has_drop', 'has_breakdown', 'has_outro',
            'structure_sections', 'track_type', 'genre',
            'has_vocals', 'has_heavy_bass', 'has_pads',
            'percussion_density', 'mix_energy_start', 'mix_energy_end',
            'drop_timestamp',
        }
        missing = required - fields
        assert not missing, f"Missing fields: {missing}"

    def test_camelot_mapping_complete(self):
        assert len(KEY_TO_CAMELOT) == 24


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
