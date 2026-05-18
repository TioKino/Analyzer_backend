"""Ranking "mejor gana" para sync comunitario (item 8 del PENDING).

Cuando dos analisis del mismo fingerprint compiten por escribirse en la
cache comunitaria, gana la fuente con mayor prioridad. Para empates,
gana el primero (estabilidad) salvo si el existente esta vacio.

Modulo standalone (sin librosa / fastapi / etc) para que el ranking sea
testeable sin levantar la app entera.
"""

from __future__ import annotations

import json
from typing import Optional


# Convencion de prioridades:
#   100+  = profesional fiable (Rekordbox, Traktor importado por XML)
#   80-99 = consenso comunitario fuerte (>=5 votos > >=3 votos)
#   60-79 = APIs comerciales (Beatport — historico, eliminado pero soportado)
#   40-59 = motor local DJ Analyzer
#   20-39 = metadata del fichero (ID3)
#   1-19  = analisis espectral generico
#   0     = pending / desconocido / cache fallback
ANALYSIS_SOURCE_PRIORITY = {
    'rekordbox': 110,
    'traktor': 105,
    'consensus_10': 95,
    'consensus_9': 94,
    'consensus_8': 93,
    'consensus_7': 92,
    'consensus_6': 91,
    'consensus_5': 90,
    'consensus_4': 85,
    'consensus_3': 80,
    'beatport': 70,
    'local_engine': 50,
    'id3': 30,
    'suggestion_2': 25,
    'analysis': 15,
    'acousticbrainz': 14,
    'discogs': 14,
    'musicbrainz': 14,
    'cached': 5,
    'pending': 0,
    '': 0,
    None: 0,
}


def get_source_priority(source) -> int:
    """Prioridad numerica de una fuente de analisis. Devuelve 0 si no la
    reconoce (defensivo: nuevos sources externos no rompen, simplemente
    pierden frente a los conocidos)."""
    if source is None:
        return 0
    if source in ANALYSIS_SOURCE_PRIORITY:
        return ANALYSIS_SOURCE_PRIORITY[source]
    if isinstance(source, str) and source.startswith('consensus_'):
        try:
            n = int(source.split('_', 1)[1])
            if n >= 3:
                return min(95, 80 + (n - 3) * 5)
            return 60
        except (ValueError, IndexError):
            return 80
    return 0


def should_overwrite_analysis(existing: Optional[dict], new_data: dict) -> bool:
    """Decide si `new_data` debe sobreescribir `existing` en la cache
    comunitaria.

    1. Sin existente → escribir.
    2. Prioridad nuevo > existente → escribir.
    3. Prioridad existente > nuevo → NO escribir.
    4. Empate y existente con BPM/key vacios → escribir.
    5. Empate con ambos validos → NO escribir (el primero gana).
    """
    if not existing:
        return True

    existing_analysis: dict = existing if isinstance(existing, dict) else {}
    ej_raw = existing_analysis.get('analysis_json')
    if ej_raw:
        try:
            parsed = json.loads(ej_raw) if isinstance(ej_raw, str) else ej_raw
            if isinstance(parsed, dict):
                existing_analysis = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    existing_source = (
        existing_analysis.get('bpm_source')
        or (existing.get('bpm_source') if isinstance(existing, dict) else None)
    )
    new_source = new_data.get('bpm_source')

    p_existing = get_source_priority(existing_source)
    p_new = get_source_priority(new_source)

    if p_new > p_existing:
        return True
    if p_new < p_existing:
        return False

    # Empate: dejar pasar si el existente esta vacio
    try:
        existing_bpm = float(
            existing_analysis.get('bpm')
            or (existing.get('bpm') if isinstance(existing, dict) else 0)
            or 0
        )
    except (TypeError, ValueError):
        existing_bpm = 0
    key_raw = (
        existing_analysis.get('key')
        or (existing.get('key') if isinstance(existing, dict) else None)
    )
    existing_key = key_raw.strip() if isinstance(key_raw, str) else ''
    if existing_bpm <= 0 or not existing_key:
        return True

    return False
