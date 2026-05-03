"""
Tests para endpoints admin nuevos (privacy-first) y persistencia de
errores de analisis (analysis_errors).

Cubre:
- /admin/users devuelve solo counts agregados, no contenido.
- /admin/users/{id}/{tracks,sessions,...} eliminados deben dar 404.
- /admin/errors-grouped agrupa por error_class+msg_short.
- log_analysis_error anonimiza filename.
- _strip_traceback_pii limpia paths con username/home.
- cleanup_old_errors purga registros viejos.
- Rate limit admin (50/min) responde 429 al pasarlo.

Para ejecutar:
    pytest test_admin_panel.py -v
"""

import os
import time
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

# IMPORTANTE: configurar el path de la BD a un tempfile ANTES de importar
# nada que use sync_endpoints, para que las tablas se creen en el tempdb.
_TMPDIR = tempfile.mkdtemp(prefix="dj_analyzer_test_")
_TEST_DB = os.path.join(_TMPDIR, "sync_test.db")
os.environ["SYNC_DB_PATH"] = _TEST_DB
os.environ["ADMIN_TOKEN"] = "test_admin_secret"

# pylint: disable=wrong-import-position
from main import app  # noqa: E402
import sync_endpoints  # noqa: E402
from sync_endpoints import (  # noqa: E402
    log_analysis_error,
    cleanup_old_errors,
    _anonymize_filename,
    _strip_traceback_pii,
)
from validation import admin_rate_limiter  # noqa: E402


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def client():
    """Cliente FastAPI con auth admin configurado."""
    return TestClient(app)


@pytest.fixture
def admin_headers():
    return {"X-Admin-Secret": "test_admin_secret"}


@pytest.fixture(autouse=True)
def reset_db():
    """Limpia tablas relevantes entre tests para aislamiento."""
    # Reset rate limiter para que tests no se afecten entre si
    admin_rate_limiter._backend._buckets.clear() if hasattr(
        admin_rate_limiter._backend, "_buckets"
    ) else None

    conn = sqlite3.connect(_TEST_DB)
    try:
        conn.execute("DELETE FROM analysis_errors")
        conn.execute("DELETE FROM sync_items")
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM user_devices")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Tablas aun no creadas en algun import-time path
    finally:
        conn.close()
    yield


# ============================================================================
# TEST: log_analysis_error + anonimizacion
# ============================================================================


class TestLogAnalysisError:
    def test_anonymize_filename_keeps_extension(self):
        out = _anonymize_filename("Bad Bunny - Featuring Real Name.mp3")
        assert out.endswith(".mp3")
        assert "Bad Bunny" not in out
        assert "Featuring" not in out
        # 8 chars hash + ".mp3" = 12
        assert len(out) == 12

    def test_anonymize_strips_path(self):
        out = _anonymize_filename(
            "C:\\Users\\juan.garcia\\Music\\unreleased.mp3"
        )
        assert "juan.garcia" not in out
        assert "Users" not in out
        assert out.endswith(".mp3")

    def test_anonymize_handles_no_extension(self):
        out = _anonymize_filename("strange_file")
        # Sin punto, no extension; sigue habiendo hash
        assert len(out) == 8
        assert "strange" not in out

    def test_anonymize_empty_returns_unknown(self):
        assert _anonymize_filename("") == "(unknown)"

    def test_strip_traceback_windows_paths(self):
        tb = (
            'File "C:\\Users\\juan\\AppData\\Local\\app\\main.py", '
            'line 1, in <module>'
        )
        out = _strip_traceback_pii(tb)
        assert "juan" not in out
        assert "<user>" in out

    def test_strip_traceback_unix_home(self):
        tb = 'File "/home/alice/projects/app/main.py", line 1'
        out = _strip_traceback_pii(tb)
        assert "alice" not in out
        assert "<home>" in out

    def test_strip_traceback_macos_home(self):
        tb = 'File "/Users/bob/Downloads/app/main.py", line 1'
        out = _strip_traceback_pii(tb)
        assert "bob" not in out
        assert "<home>" in out

    def test_strip_traceback_tempfile(self):
        tb = "Cannot read /tmp/tmpABC123.mp3 - corrupted"
        out = _strip_traceback_pii(tb)
        assert "tmpABC123" not in out
        assert "<tmpfile>" in out

    def test_log_analysis_error_persists_anonymized(self):
        rid = log_analysis_error(
            filename="my secret track.mp3",
            error_class="TypeError",
            error_msg="example",
            device_id="dja_test",
            traceback_str='File "C:\\Users\\dev\\code\\main.py", line 99',
        )
        assert rid is not None

        conn = sqlite3.connect(_TEST_DB)
        try:
            row = conn.execute(
                "SELECT filename, traceback FROM analysis_errors WHERE id = ?",
                (rid,),
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        filename, traceback = row
        assert "secret" not in filename
        assert filename.endswith(".mp3")
        assert "dev" not in traceback
        assert "<user>" in traceback


# ============================================================================
# TEST: cleanup_old_errors
# ============================================================================


class TestCleanup:
    def test_cleanup_removes_old_resolved(self):
        # Insertar 1 resuelto viejo + 1 reciente
        from datetime import datetime, timezone, timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        new = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(_TEST_DB)
        try:
            conn.execute(
                """INSERT INTO analysis_errors
                   (timestamp, filename, error_class, error_msg, resolved, endpoint)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (old, "x.mp3", "T", "old", 1, "/analyze"),
            )
            conn.execute(
                """INSERT INTO analysis_errors
                   (timestamp, filename, error_class, error_msg, resolved, endpoint)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (new, "y.mp3", "T", "new", 1, "/analyze"),
            )
            conn.commit()

            deleted = cleanup_old_errors(
                resolved_max_age_days=30, unresolved_max_age_days=90
            )
            assert deleted == 1

            remaining = conn.execute(
                "SELECT error_msg FROM analysis_errors"
            ).fetchall()
            assert len(remaining) == 1
            assert remaining[0][0] == "new"
        finally:
            conn.close()

    def test_cleanup_keeps_recent_unresolved(self):
        from datetime import datetime, timezone, timedelta
        # Sin resolver y reciente: NO se borra
        ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        conn = sqlite3.connect(_TEST_DB)
        try:
            conn.execute(
                """INSERT INTO analysis_errors
                   (timestamp, filename, error_class, error_msg, resolved, endpoint)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ts, "x.mp3", "T", "still bug", 0, "/analyze"),
            )
            conn.commit()
            deleted = cleanup_old_errors()
            assert deleted == 0
        finally:
            conn.close()


# ============================================================================
# TEST: endpoints admin (auth + privacy)
# ============================================================================


class TestAdminAuth:
    def test_endpoints_require_secret(self, client):
        resp = client.get("/admin/users")
        assert resp.status_code == 401

    def test_endpoints_reject_wrong_secret(self, client):
        resp = client.get(
            "/admin/users", headers={"X-Admin-Secret": "wrong"}
        )
        assert resp.status_code == 401

    def test_endpoints_accept_correct_secret(self, client, admin_headers):
        resp = client.get("/admin/users", headers=admin_headers)
        assert resp.status_code == 200


class TestAdminEndpointsExist:
    """Endpoints que SI deben existir tras el refactor privacy-first."""

    def test_users_endpoint(self, client, admin_headers):
        resp = client.get("/admin/users", headers=admin_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "users" in body and "total" in body

    def test_user_summary_endpoint(self, client, admin_headers):
        resp = client.get(
            "/admin/users/dja_test/summary", headers=admin_headers
        )
        assert resp.status_code == 200
        assert "counts" in resp.json()

    def test_user_errors_endpoint(self, client, admin_headers):
        resp = client.get(
            "/admin/users/dja_test/errors", headers=admin_headers
        )
        assert resp.status_code == 200
        assert "errors" in resp.json()

    def test_errors_endpoint(self, client, admin_headers):
        resp = client.get("/admin/errors", headers=admin_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "errors" in body
        assert "unresolved_total" in body

    def test_errors_grouped_endpoint(self, client, admin_headers):
        resp = client.get("/admin/errors-grouped", headers=admin_headers)
        assert resp.status_code == 200
        assert "groups" in resp.json()

    def test_stats_endpoint(self, client, admin_headers):
        resp = client.get("/admin/stats", headers=admin_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "total_users" in body
        assert "errors_unresolved" in body


class TestAdminEndpointsRemoved:
    """Endpoints eliminados deliberadamente (filtraban contenido sensible)
    deben devolver 404 — no estan registrados en el router."""

    @pytest.mark.parametrize("path", [
        "/admin/users/dja_test/tracks",
        "/admin/users/dja_test/previews",
        "/admin/users/dja_test/sessions",
        "/admin/users/dja_test/folders",
        "/admin/users/dja_test/collections",
        "/admin/users/dja_test/cues",
        "/admin/users/dja_test/favorites",
        "/admin/users/dja_test/overrides",
        "/admin/all-tracks",
        "/admin/all-previews",
    ])
    def test_removed_endpoint_returns_404(self, client, admin_headers, path):
        resp = client.get(path, headers=admin_headers)
        assert resp.status_code == 404, (
            f"{path} debe estar eliminado pero devolvio {resp.status_code}"
        )


# ============================================================================
# TEST: errors-grouped agrupa correctamente
# ============================================================================


class TestErrorsGrouped:
    def test_groups_by_class_and_msg(self, client, admin_headers):
        # 3 errores del mismo tipo + 1 distinto
        for i in range(3):
            log_analysis_error(
                filename=f"track{i}.mp3",
                error_class="TypeError",
                error_msg="only 0-dimensional arrays",
                device_id=f"dja_user{i}",
            )
        log_analysis_error(
            filename="other.mp3",
            error_class="ValidationError",
            error_msg="genre None",
            device_id="dja_user_x",
        )

        resp = client.get(
            "/admin/errors-grouped", headers=admin_headers
        )
        assert resp.status_code == 200
        groups = resp.json()["groups"]
        # 2 grupos: TypeError y ValidationError
        assert len(groups) == 2
        type_err = next(g for g in groups if g["error_class"] == "TypeError")
        assert type_err["count"] == 3
        assert type_err["devices_affected"] == 3


# ============================================================================
# TEST: rate limit admin (50/min)
# ============================================================================


class TestAdminRateLimit:
    def test_rate_limit_kicks_in(self, client, admin_headers):
        # Hacer 60 requests rapidas; las primeras 50 OK, despues 429.
        ok_count = 0
        rate_limited = False
        for _ in range(60):
            resp = client.get("/admin/users", headers=admin_headers)
            if resp.status_code == 200:
                ok_count += 1
            elif resp.status_code == 429:
                rate_limited = True
                break
        assert ok_count <= 50
        assert rate_limited, "Rate limit deberia haber disparado"
