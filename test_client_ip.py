"""
Tests de la resolucion de IP del cliente desde X-Forwarded-For.

SEGURIDAD: el rate-limit se agrupa por IP. Si la IP se toma de la parte
IZQUIERDA de X-Forwarded-For (controlable por el cliente), cualquiera manda un
XFF distinto por request y se salta el rate-limit. La entrada FIABLE es la que
añade nuestro proxy de confianza (Render): la `trusted_hops`-esima desde la
derecha. Estos tests fijan ese contrato.

Importa `validation` (que arrastra fastapi), asi que corre en CI igual que
test_validation.py.
"""

import pytest

from validation import _pick_client_ip


# ── Render estandar: 1 proxy de confianza ────────────────────────────

def test_clean_single_ip():
    assert _pick_client_ip("203.0.113.7", "proxyhost", 1) == "203.0.113.7"


def test_spoofed_xff_is_ignored():
    """Cliente manda 9.9.9.9; Render anexa la IP real por la derecha. Tomamos
    la real, NO la falseada."""
    assert _pick_client_ip("9.9.9.9, 203.0.113.7", "ph", 1) == "203.0.113.7"


def test_multiple_spoofed_entries_still_ignored():
    """Aunque el atacante meta varias IPs falsas para empujar la real, la
    ultima (la que pone Render) sigue ganando."""
    assert _pick_client_ip("1.1.1.1, 2.2.2.2, 203.0.113.7", "ph", 1) == "203.0.113.7"


# ── 2 proxies de confianza (p.ej. Cloudflare -> Render) ──────────────

def test_two_trusted_hops():
    # cadena: spoof, REALCLIENT (anexa CF), cf-ip (anexa Render)
    assert _pick_client_ip("spoof, 203.0.113.7, cf-ip", "ph", 2) == "203.0.113.7"


def test_two_hops_with_extra_spoof_prefix():
    assert _pick_client_ip("a, b, 203.0.113.7, cf-ip", "ph", 2) == "203.0.113.7"


# ── Fallbacks ────────────────────────────────────────────────────────

def test_no_xff_falls_back_to_client_host():
    assert _pick_client_ip(None, "198.51.100.5", 1) == "198.51.100.5"


def test_blank_xff_falls_back_to_client_host():
    assert _pick_client_ip("  ,  , ", "198.51.100.5", 1) == "198.51.100.5"


def test_nothing_known_returns_unknown():
    assert _pick_client_ip(None, None, 1) == "unknown"


def test_more_configured_hops_than_present_clamps_left():
    """Si la cadena es mas corta que los hops configurados, no petar: caer a la
    entrada mas a la izquierda disponible."""
    assert _pick_client_ip("203.0.113.7", "ph", 3) == "203.0.113.7"
