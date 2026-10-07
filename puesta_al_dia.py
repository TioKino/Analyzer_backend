"""
La PUESTA AL DÍA de lo ya analizado (2026-10-07).

El DSP cambió sin subir `ANALYSIS_VERSION` (decisión owner): la energía de los
temas de más de 4 minutos analizados en Render salía un nivel alta (#5) y
*closing* salía por tener outro (#7). Lo ya analizado se quedaba como estaba.
Rehacerlo con «Reanalizar» borra las ediciones del DJ y sube cada tema entero,
así que se rehace en SEGUNDO PLANO desde el ordenador, solo lo que cambia:

- ``closing``: el tipo salió *closing* con la regla vieja.
- ``trozos``: más de 4 minutos y NO lo analizó un motor local, o sea que pasó
  por el camino por trozos de Render (energía, tipo, género y graves).

Quién decide qué entra es ESTE módulo, no el cliente: el cliente no sabe qué
motor analizó cada tema ni cuándo lo analizó Render, y así la regla se cambia
sin release.

Interruptor en Render: ``PUESTA_AL_DIA`` es la fecha de corte (``YYYY-MM-DD``,
la del deploy del DSP nuevo). Sin ella está APAGADA y nadie rehace nada. Lo
analizado desde esa fecha ya es del DSP nuevo. ``PUESTA_AL_DIA_TOPE_RENDER``
es cuántos temas al día puede mandar a Render un aparato sin motor local (el
Mac App Store), que solo rehace los ``closing``.
"""

import os
import re
from datetime import date
from typing import Dict, List, Optional

# El de `main.CHUNK_ANALYSIS_THRESHOLD`: lo que dura más que esto pasó por el
# camino por trozos en Render. Un test exige que sean el mismo número.
UMBRAL_TROZOS = 240

TOPE_RENDER_POR_DEFECTO = 20
TOPE_RENDER_MAXIMO = 200

# Columnas que hacen falta para decidir; sin `analysis_json` (pesa ~2 KB por
# fila y aquí no se lee). El fallback de un análisis fallido se reconoce por
# el BPM a 0: `analysis_status` vive dentro del JSON, no es columna.
COLUMNAS = ('id, fingerprint, analyzed_at, engine_source, duration, '
            'track_type, bpm')


def corte() -> Optional[str]:
    """La fecha de corte, o None si la puesta al día está apagada."""
    v = (os.environ.get('PUESTA_AL_DIA') or '').strip()
    return v if re.match(r'^\d{4}-\d{2}-\d{2}', v) else None


def tope_render() -> int:
    """Temas al día que un aparato sin motor local puede mandar a Render."""
    try:
        n = int((os.environ.get('PUESTA_AL_DIA_TOPE_RENDER') or '').strip())
    except ValueError:
        return TOPE_RENDER_POR_DEFECTO
    return max(0, min(n, TOPE_RENDER_MAXIMO))


def motivos(fila: Dict, desde: str, umbral: float = UMBRAL_TROZOS) -> List[str]:
    """Por qué hay que rehacer esta fila (vacío = nada).

    Solo lo analizado ANTES de `desde`; un análisis fallido (bpm 0) no se
    rehace: el DSP ya no pudo con ese fichero."""
    if fila.get('analysis_status') == 'failed':
        return []
    try:
        if float(fila.get('bpm') or 0) <= 0:
            return []
    except (TypeError, ValueError):
        return []
    cuando = fila.get('analyzed_at') or ''
    if cuando and cuando >= desde:
        return []
    salida = []
    if (fila.get('track_type') or '').strip().lower() == 'closing':
        salida.append('closing')
    try:
        dura = float(fila.get('duration') or 0)
    except (TypeError, ValueError):
        dura = 0.0
    if dura > umbral and fila.get('engine_source') != 'local_engine':
        salida.append('trozos')
    return salida


class TopeDiario:
    """Cuántos temas lleva hoy cada aparato en Render. En memoria: un deploy
    lo pone a cero, y es solo la red de seguridad (el cliente ya respeta el
    tope que le da `/puesta-al-dia/candidatos`)."""

    def __init__(self):
        self._dia = None
        self._cuenta: Dict[str, int] = {}

    def apuntar(self, aparato: str, tope: int, hoy: Optional[date] = None) -> bool:
        """Apunta uno más y dice si cabe."""
        hoy = hoy or date.today()
        if self._dia != hoy:
            self._dia, self._cuenta = hoy, {}
        n = self._cuenta.get(aparato, 0)
        if n >= tope:
            return False
        self._cuenta[aparato] = n + 1
        return True
