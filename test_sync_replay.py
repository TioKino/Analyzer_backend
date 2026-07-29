"""Tests para el anti-replay del HMAC de sync (`_verify_sync_auth`).

Cubre la fase 1 (backward-compatible) del rollout:
- Cliente NUEVO firma "<ts>.<body>" + manda X-Timestamp -> se verifica frescura.
- Cliente VIEJO firma solo "<body>" sin X-Timestamp -> se acepta (sin anti-replay).

Es codigo de auth critico (protege /sync/*), asi que aqui se blinda cada rama:
firma valida, firma invalida, replay (timestamp viejo/futuro), reutilizacion de
firma sobre otro body, rotacion de secret y modo dev sin secret.
"""

import asyncio
import hashlib
import hmac as _hmac
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

import sync_endpoints

SECRET = "test-sync-secret"


def _now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _sign(secret, payload: bytes) -> str:
    return _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _make_request(body: bytes, headers: dict):
    """Construye un starlette Request minimo con body + headers dados."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/sync/push",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _verify(body: bytes, headers: dict):
    """Corre la dependency y devuelve None si pasa; propaga HTTPException si no."""
    return asyncio.run(sync_endpoints._verify_sync_auth(_make_request(body, headers)))


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setattr(sync_endpoints, "SYNC_AUTH_SECRET", SECRET)
    return SECRET


class TestNewStyleSignedRequest:
    def test_valid_timestamped_signature_passes(self, secret):
        body = b'{"device_id":"d1"}'
        ts = str(_now_ms())
        sig = _sign(SECRET, ts.encode() + b"." + body)
        # No debe lanzar.
        assert _verify(body, {"X-Timestamp": ts, "X-Signature": sig}) is None

    def test_stale_timestamp_rejected(self, secret):
        body = b'{"device_id":"d1"}'
        # Fuera de la ventana (3600s por defecto) -> replay.
        ts = str(_now_ms() - (sync_endpoints.SYNC_REPLAY_WINDOW_SEC + 60) * 1000)
        sig = _sign(SECRET, ts.encode() + b"." + body)
        with pytest.raises(HTTPException) as exc:
            _verify(body, {"X-Timestamp": ts, "X-Signature": sig})
        assert exc.value.status_code == 401
        assert "timestamp" in exc.value.detail.lower()

    def test_future_timestamp_beyond_window_rejected(self, secret):
        body = b'{"device_id":"d1"}'
        ts = str(_now_ms() + (sync_endpoints.SYNC_REPLAY_WINDOW_SEC + 60) * 1000)
        sig = _sign(SECRET, ts.encode() + b"." + body)
        with pytest.raises(HTTPException) as exc:
            _verify(body, {"X-Timestamp": ts, "X-Signature": sig})
        assert exc.value.status_code == 401

    def test_non_numeric_timestamp_rejected(self, secret):
        body = b'{"device_id":"d1"}'
        sig = _sign(SECRET, b"abc." + body)
        with pytest.raises(HTTPException) as exc:
            _verify(body, {"X-Timestamp": "abc", "X-Signature": sig})
        assert exc.value.status_code == 401

    def test_captured_signature_replayed_on_other_body_rejected(self, secret):
        """Firma capturada de body A, reusada con ts fresco sobre body B -> falla."""
        body_a = b'{"device_id":"d1","amount":1}'
        ts = str(_now_ms())
        sig_a = _sign(SECRET, ts.encode() + b"." + body_a)
        body_b = b'{"device_id":"d1","amount":9999}'
        with pytest.raises(HTTPException) as exc:
            _verify(body_b, {"X-Timestamp": ts, "X-Signature": sig_a})
        assert exc.value.status_code == 401


class TestLegacyBackwardCompat:
    def test_body_only_signature_still_accepted(self, secret):
        """Cliente viejo sin X-Timestamp firma solo el body -> se acepta (fase 1)."""
        body = b'{"device_id":"d1"}'
        sig = _sign(SECRET, body)
        assert _verify(body, {"X-Signature": sig}) is None

    def test_legacy_invalid_signature_rejected(self, secret):
        body = b'{"device_id":"d1"}'
        with pytest.raises(HTTPException) as exc:
            _verify(body, {"X-Signature": "deadbeef"})
        assert exc.value.status_code == 401


class TestMissingAndRotation:
    def test_missing_signature_header_rejected(self, secret):
        with pytest.raises(HTTPException) as exc:
            _verify(b"{}", {})
        assert exc.value.status_code == 401
        assert "signature" in exc.value.detail.lower()

    def test_secret_rotation_old_secret_still_valid(self, monkeypatch):
        """SYNC_AUTH_SECRET='nuevo,viejo': una firma con el viejo sigue pasando."""
        monkeypatch.setattr(sync_endpoints, "SYNC_AUTH_SECRET", "new-secret,old-secret")
        body = b'{"device_id":"d1"}'
        ts = str(_now_ms())
        sig_old = _sign("old-secret", ts.encode() + b"." + body)
        assert _verify(body, {"X-Timestamp": ts, "X-Signature": sig_old}) is None

    def test_secret_rotation_new_secret_valid(self, monkeypatch):
        monkeypatch.setattr(sync_endpoints, "SYNC_AUTH_SECRET", "new-secret,old-secret")
        body = b'{"device_id":"d1"}'
        ts = str(_now_ms())
        sig_new = _sign("new-secret", ts.encode() + b"." + body)
        assert _verify(body, {"X-Timestamp": ts, "X-Signature": sig_new}) is None


class TestDevModeNoSecret:
    def test_no_secret_local_dev_allows_request(self, monkeypatch):
        """Sin SYNC_AUTH_SECRET y sin RENDER/RAILWAY -> modo dev local, pasa."""
        monkeypatch.setattr(sync_endpoints, "SYNC_AUTH_SECRET", "")
        monkeypatch.delenv("RENDER", raising=False)
        monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
        assert _verify(b"{}", {}) is None

    def test_no_secret_in_production_raises_500(self, monkeypatch):
        """Sin secret pero en Render -> error de config (no se sirve sin auth)."""
        monkeypatch.setattr(sync_endpoints, "SYNC_AUTH_SECRET", "")
        monkeypatch.setenv("RENDER", "true")
        with pytest.raises(HTTPException) as exc:
            _verify(b"{}", {})
        assert exc.value.status_code == 500
