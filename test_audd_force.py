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


# ============================================================================
# EL MOTIVO IMPORTA, NO SOLO EL SI/NO
# ============================================================================
# Reportado como «pulso Limpiar con AudD y no funciona». No era el cupo: ese
# dia iban 44 llamadas de 100. Lo que pasaba es que `force` NO salta el corte
# por duracion —y hace bien, AudD cobra por llamada y una sesion de una hora no
# se identifica ni gastandola— pero ese skip se loguea en DEBUG, o sea invisible
# en Render, y el cliente lo mostraba como «no se pudo identificar (o se agoto
# la cuota diaria)»: dos causas distintas en una frase, y ninguna cierta.
#
# `/analyze` devuelve ahora `audd_skipped_reason` con el motivo tal cual, para
# que la UI pueda decir la verdad. Estos tests fijan que el motivo se distinga.

def test_force_no_salta_el_corte_por_duracion():
    # Un megamix de 68 min. Da igual que el usuario lo fuerce: AudD no lo va a
    # identificar, asi que gastar la llamada seria tirar dinero.
    ok, reason = should_trigger_audd(
        "Unknown", "Track 5", 4080.0, "fp_mix", FakeDB(),
        max_duration=720.0, force=True,
    )
    assert ok is False
    assert reason.startswith("duracion>"), reason


def test_el_motivo_distingue_duracion_de_cupo():
    # Los dos acaban en «AudD no corrio», pero uno se arregla mañana y el otro
    # no se arregla nunca. Meterlos en el mismo mensaje es lo que hacia que la
    # feature pareciera rota.
    _, por_duracion = should_trigger_audd(
        "Unknown", "Track 5", 4080.0, "fp_mix", FakeDB(), force=True,
    )
    _, por_cupo = should_trigger_audd(
        "Unknown", "Track 5", 240.0, "fp_corto",
        FakeDB(today_count=100), daily_cap=100, force=True,
    )

    assert por_duracion.startswith("duracion>")
    assert por_cupo.startswith("daily cap")
    assert por_duracion != por_cupo


def test_un_track_de_duracion_normal_forzado_SI_dispara():
    # El seguro del seguro: que acotar el motivo no se lleve por delante el
    # caso que si tiene que funcionar.
    ok, _ = should_trigger_audd(
        "Unknown", "Track 5", 240.0, "fp_corto", FakeDB(today_count=44),
        daily_cap=100, force=True,
    )
    assert ok is True
