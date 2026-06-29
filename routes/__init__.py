"""
Route modules for DJ Analyzer Pro API.

NOTA (cleanup review 2026-06-29): los módulos search/library/community/preview/
media se BORRARON. Eran routers MUERTOS (definidos pero nunca montados:
`init_all` no se llamaba en ningún sitio y ningún `include_router` los registraba)
y duplicaban endpoints que viven inline en `main.py`. Esa duplicación stale fue
exactamente la causa del doble incidente de `/admin/reset-database`. El único
módulo de ruta VIVO es `admin_panel.py`, que `main.py` importa directamente como
submódulo (`from routes.admin_panel import admin_panel_router`), así que este
`__init__` no necesita re-exportar nada.
"""
