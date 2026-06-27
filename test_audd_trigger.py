"""
Tests de `should_trigger_audd` — el guard de GASTO de AudD (cobra por llamada).

Foco principal: el control de coste debe ser FAIL-CLOSED. Si la BD falla al
comprobar el cooldown o el daily cap, NO se dispara AudD (antes caia a
`return True` y un lock/timeout de BD saltaba el tope de gasto).

`should_trigger_audd` no importa librosa a nivel modulo (librosa es lazy dentro
de `call_audd`), asi que estos tests corren en cualquier entorno.
"""

import pytest

from audd_helper import should_trigger_audd


class FakeDB:
    def __init__(self, last_call=None, today_count=0,
                 cooldown_raises=False, cap_raises=False):
        self._last_call = last_call
        self._today = today_count
        self._cooldown_raises = cooldown_raises
        self._cap_raises = cap_raises

    def get_last_audd_call(self, fingerprint):
        if self._cooldown_raises:
            raise RuntimeError("db locked (cooldown)")
        return self._last_call

    def count_audd_calls_today(self):
        if self._cap_raises:
            raise RuntimeError("db locked (cap)")
        return self._today


GARBAGE = (None, None)  # artist/title ausentes -> garbage metadata


# ── Decision normal ──────────────────────────────────────────────────

def test_garbage_under_cap_fires():
    fire, _ = should_trigger_audd(*GARBAGE, 120, "fp", FakeDB(), daily_cap=50)
    assert fire is True


def test_usable_metadata_does_not_fire():
    fire, reason = should_trigger_audd("Oxia", "Domino", 120, "fp", FakeDB())
    assert fire is False
    assert "utilizable" in reason


def test_duration_too_short():
    fire, _ = should_trigger_audd(*GARBAGE, 10, "fp", FakeDB(), min_duration=30)
    assert fire is False


def test_duration_too_long():
    fire, _ = should_trigger_audd(*GARBAGE, 9999, "fp", FakeDB(), max_duration=720)
    assert fire is False


def test_cap_reached_blocks():
    fire, reason = should_trigger_audd(
        *GARBAGE, 120, "fp", FakeDB(today_count=50), daily_cap=50)
    assert fire is False
    assert "cap" in reason


def test_cooldown_active_blocks():
    import time
    recent = time.time() - 2 * 86400  # hace 2 dias
    fire, reason = should_trigger_audd(
        *GARBAGE, 120, "fp", FakeDB(last_call=recent), cooldown_days=7)
    assert fire is False
    assert "cooldown" in reason


# ── FAIL-CLOSED: un fallo de BD NO concede permiso para gastar ───────

def test_cap_query_failure_is_fail_closed():
    fire, reason = should_trigger_audd(
        *GARBAGE, 120, "fp", FakeDB(cap_raises=True), daily_cap=50)
    assert fire is False
    assert "cap check error" in reason


def test_cooldown_query_failure_is_fail_closed():
    fire, reason = should_trigger_audd(
        *GARBAGE, 120, "fp", FakeDB(cooldown_raises=True), cooldown_days=7)
    assert fire is False
    assert "cooldown check error" in reason


# ── force: salta metadata y cooldown, pero el cap sigue mandando ─────

def test_force_skips_cooldown_check_entirely():
    """force=True ni entra al bloque de cooldown, asi que un fallo de esa query
    no lo afecta; con cap OK, dispara."""
    fire, _ = should_trigger_audd(
        "Oxia", "Domino", 120, "fp",
        FakeDB(cooldown_raises=True, today_count=0), force=True, daily_cap=50)
    assert fire is True


def test_force_still_respects_cap_failure():
    """Incluso con force, si el cap no se puede verificar -> fail-closed."""
    fire, reason = should_trigger_audd(
        "Oxia", "Domino", 120, "fp",
        FakeDB(cap_raises=True), force=True, daily_cap=50)
    assert fire is False
    assert "cap check error" in reason
