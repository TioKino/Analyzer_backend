"""
Route modules for DJ Analyzer Pro API.

ESTRUCTURA (troceo de main.py, review 2026-06-29). main.py paso de ~5100 a
~4083 lineas extrayendo los clusters de endpoints HTTP cohesivos a modulos de
ruta. Patron uniforme en todos: cada modulo define su(s) APIRouter, expone un
`init(...)` que INYECTA las dependencias que antes eran globales de main.py
(la instancia de BD, paths de cache, helpers compartidos) y se monta en main
con `init_X(...)` + `app.include_router(...)`. Los endpoints inline se BORRARON
en el mismo commit -> NO hay duplicacion (la duplicacion stale fue justo la
causa del doble incidente de `/admin/reset-database`).

Modulos VIVOS (todos montados via include_router en main.py):
  - admin_panel.py      /admin/* (panel read-only Flutter). Importado directo.
  - search.py           /search/*, /search-analyzed, /library/*, /track/{id}   (paso 2, #29)
  - community.py        /community/* (beat-grid, overrides, notes, ratings,
                        popularity)                                            (paso 3, #30)
  - preview.py          /preview/{id}, /previews/check, /artworks/check,
                        /preview/upload/{id}                                   (paso 4, #31)
  - analysis_artwork.py /check-analyzed[-by-fingerprint], /analysis/*,
                        /artwork/* (HEAD/GET/upload)                           (paso 5, #32)

QUE SE QUEDA INLINE EN main.py (decision firme 2026-06-29 — NO trocear):
  - Nucleo de analisis: /analyze, /identify, /recognize, /cache-analysis,
    /correction + sus ~15 helpers (try_bpm_double_half, search_collective_db,
    _is_analysis_current, enriquecimiento AudD/artwork...). Es UNA unidad
    fuertemente acoplada (la logica DSP real); trocearla no es un code-move
    mecanico sino cirugia del motor. Si algun dia se hace, extraer PRIMERO los
    helpers puros a analysis_core.py, no los endpoints de golpe.
  - Admin destructivo: /admin/reset-database, /admin/clear-artwork-cache —
    sensibles, mejor a la vista en main.py.
  - Meta: /, /announcement, /health, /client-error — triviales, mover = ruido.

HISTORICO: los modulos search/library/community/preview/media originales (otros,
borrados en #28) eran routers MUERTOS — definidos pero nunca montados (`init_all`
no se llamaba y ningun include_router los registraba) y duplicaban los inline.
Los modulos vivos de arriba son reescrituras nuevas que SI se montan.
"""
