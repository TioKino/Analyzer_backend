"""
Tests del bypass `force=True` en AudD auto-trigger.

Cubre el contrato del parametro `force_audd` que /analyze acepta cuando el
usuario pide "limpiar con AudD" desde la UI: saltea el cooldown 7d y el
check de garbage metadata, pero sigue respetando el daily cap y los
limites de duracion.
"""
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audd_helper import should_trigger_audd


class FakeDB:
    """Stub minimo que imita la API de db que should_trigger_audd consume."""

    def __init__(self, last_call=None, today_count=0):
        self._last_call = last_call
        self._today_count = today_count

    def get_last_audd_call(self, fingerprint):
        return self._last_call

    def count_audd_calls_today(self):
        return self._today_count


def _now_minus_days(days):
    return datetime.now(timezone.utc).timestamp() - days * 86400


def test_force_bypass_garbage_metadata_check():
    # Sin force: metadata utilizable -> no dispara.
    ok, reason = should_trigger_audd(
        "Daft Punk", "One More Time", 240.0, "fp123", FakeDB(),
    )
    assert ok is False
    assert "utilizable" in reason

    # Con force: metadata utilizable -> dispara igualmente.
    ok, reason = should_trigger_audd(
        "Daft Punk", "One More Time", 240.0, "fp123", FakeDB(),
        force=True,
    )
    assert ok is True
    assert "force" in reason


def test_force_bypass_cooldown():
    # 3d desde el ultimo intento, cooldown 7d -> sin force, skip.
    db = FakeDB(last_call=_now_minus_days(3))
    ok, reason = should_trigger_audd(
        None, None, 240.0, "fp123", db, cooldown_days=7,
    )
    assert ok is False
    assert "cooldown" in reason

    # Mismo escenario con force -> dispara.
    ok, reason = should_trigger_audd(
        None, None, 240.0, "fp123", db, cooldown_days=7, force=True,
    )
    assert ok is True


def test_force_respects_daily_cap():
    # Daily cap alcanzado -> ni siquiera con force se dispara (cuota dura).
    db = FakeDB(today_count=50)
    ok, reason = should_trigger_audd(
        None, None, 240.0, "fp123", db, daily_cap=50, force=True,
    )
    assert ok is False
    assert "daily cap" in reason


def test_force_respects_duration_bounds():
    # Track demasiado corto -> force no aplica, fragmento no seria valido.
    ok, reason = should_trigger_audd(
        None, None, 10.0, "fp123", FakeDB(),
        min_duration=30.0, force=True,
    )
    assert ok is False
    assert "duracion" in reason

    # Track demasiado largo -> tampoco.
    ok, reason = should_trigger_audd(
        None, None, 1200.0, "fp123", FakeDB(),
        max_duration=720.0, force=True,
    )
    assert ok is False
    assert "duracion" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
