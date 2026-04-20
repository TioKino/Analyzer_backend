# DJ Analyzer Pro - Audit Report
## Date: 2026-04-11 | Scope: Backend (Python/FastAPI) + Frontend (Flutter/Dart)

---

## EXECUTIVE SUMMARY

Full code audit across both repositories. **17 bugs fixed**, **38+ issues documented** across critical, high, medium, and low severity. The most impactful fixes were:
- **Backend endpoint completely broken** (`/search/compatible/{camelot}` - passed list instead of string)
- **SQL queries on non-existent columns** (`search_collective_db` - caused runtime crashes)
- **Missing database table** (`community_cues` - all community cue operations failed)
- **Pydantic model crash on extra keys** (cached results with extra fields caused 500 errors)
- **CORS blocking sync headers** (X-Signature, X-Device-Id not allowed)
- **Syntax error** (curly quotes in `essentia_analyzer.py` prevented import)

---

## BUGS FIXED (17 total)

### Backend Fixes (12)

| # | Severity | File | Bug | Fix |
|---|----------|------|-----|-----|
| 1 | **CRITICAL** | `routes/search.py:127` | `search_compatible_keys()` received a list instead of a string, crashing the endpoint | Pass `camelot` string, not the pre-computed `compatible` list |
| 2 | **CRITICAL** | `main.py:252-267` | `search_collective_db` queried `label`, `bpm_source`, `key_source` columns that don't exist in the `tracks` table | Query only existing columns; extract extra fields from `analysis_json` |
| 3 | **CRITICAL** | `main.py:206` | `AnalysisDB()` called without path, ignoring `config.DATABASE_PATH` | Pass `db_path=DATABASE_PATH` from config |
| 4 | **CRITICAL** | `database.py` | `community_cues` table never created, all community cue operations failed | Added CREATE TABLE in `init_db()` |
| 5 | **CRITICAL** | `models.py` | `AnalysisResult(**analysis_json)` crashed when JSON had extra keys (e.g. `original_file_path`) | Added `model_config = {"extra": "ignore"}` |
| 6 | **CRITICAL** | `essentia_analyzer.py` | Curly quotes (Unicode U+201C/U+201D) in f-strings caused SyntaxError | Replaced all curly quotes with straight quotes |
| 7 | **HIGH** | `main.py:671` | `'track_type' not in dir()` always evaluates False (dir() returns module names) | Changed to `try/except NameError` pattern |
| 8 | **HIGH** | `main.py:190` | CORS `allow_headers` missing `X-Signature`, `X-Device-Id`, `X-Original-Path` | Added all required headers |
| 9 | **HIGH** | `main.py:1009` | `json.loads(existing[11])` crashed when `existing[11]` was `None` | Added `and existing[11]` guard |
| 10 | **HIGH** | `validation.py:54-58` | SQL injection regex blocked legitimate names ("Drop The Mic", "Guns N' Roses") | Changed to multi-keyword patterns; allowed single quotes |
| 11 | **HIGH** | `sync_endpoints.py:883-906` | `/sync/clear` accepted any non-empty `X-Admin-Key` header (no comparison to actual token) | Now validates `X-Admin-Key == ADMIN_TOKEN` |
| 12 | **MEDIUM** | `conftest.py:10` | `sys.path.insert` added parent directory instead of project directory | Fixed to project directory |

Additional backend fixes:
- `database.py`: Added WAL mode to persistent connection for concurrency safety
- `database.py`: `_row_to_dict` now includes `chromaprint` column (index 14)
- `main.py`: Removed duplicate import of `classify_genre_advanced`

### Frontend Fixes (5)

| # | Severity | File | Bug | Fix |
|---|----------|------|-----|-----|
| 1 | **HIGH** | `listen_screen.dart:1777` | Track type `peak_time` displayed as "Peak_time" instead of "Peak Time" | Split on underscore and capitalize each word |
| 2 | **HIGH** | `listen_screen.dart:62-65` | `AudioRecognitionService` (holds microphone) never disposed | Added `_recognitionService.dispose()` in `dispose()` |
| 3 | **HIGH** | `cue_pair.dart`, `track_cue.dart`, `library_folder.dart`, `suggest_history.dart` | `DateTime.parse()` throws on malformed dates from corrupted JSON | Changed to `DateTime.tryParse() ?? DateTime.now()` |
| 4 | **MEDIUM** | `audio_player_service.dart:36` | Only stripped `.mp3` and `.wav` extensions from title, missing `.flac`, `.m4a`, etc. | Changed to regex `r'\.[^.]+$'` to strip any extension |
| 5 | **MEDIUM** | `saved_session.dart:59-63` | Session duration used hardcoded 6.5 min/track instead of actual track durations | Now sums actual `track.duration` values |

---

## KNOWN ISSUES NOT FIXED (Require Manual Testing / Design Decisions)

### Backend - Remaining Issues

| # | Severity | Issue | Reason Not Fixed |
|---|----------|-------|------------------|
| 1 | MEDIUM | `has_heavy_bass` detection measures first 10s amplitude, not bass frequency | Requires spectral analysis redesign |
| 2 | MEDIUM | Beatport `track_type_source` overwritten to 'waveform' at line 669 | Logic flow issue - needs architectural review |
| 3 | MEDIUM | CORS allows wildcard + credentials in DEBUG mode (browsers reject this combo) | Intentional for dev mode |
| 4 | LOW | `spectral_genre_classifier.py` has overlapping BPM ranges (House/Techno at 120-130) | Genre classification heuristic - needs testing |
| 5 | LOW | `artwork_and_cuepoints.py` hardcodes `/data/artwork_cache` instead of using config | Would need import cycle resolution |
| 6 | LOW | `search_analyzed_endpoint.py` and `endpoints_adicionales.py` are dead code (never imported) | May be intentionally kept for reference |
| 7 | LOW | `test_beatport.py` is a runnable script, not a proper test | Should be restructured as pytest test |

### Frontend - Remaining Issues

| # | Severity | Issue | Reason Not Fixed |
|---|----------|-------|------------------|
| 1 | HIGH | "NOT IN LIBRARY" false positives (SHA1 vs MD5 ID mismatch) | Architectural: needs Chromaprint migration on both sides |
| 2 | HIGH | `ListenScreen` uses legacy `listen_history` SharedPreferences key while provider uses `listen_history_unified` | Needs data migration strategy |
| 3 | HIGH | `FavoritesService.load()` never called during initialization | Need to verify initialization chain |
| 4 | MEDIUM | `previewPlayer` global AudioPlayer never disposed | Global singleton - needs lifecycle management |
| 5 | MEDIUM | `persistence.dart` dual state management (globals vs Riverpod) | Legacy code - needs full refactor |
| 6 | MEDIUM | Multiple files use `Platform.isX` directly instead of `PlatformUtils` | Would break if compiled for web |
| 7 | MEDIUM | `print()` used instead of `debugPrint()` in ~27 locations | Cosmetic but affects release performance |
| 8 | LOW | Hardcoded Spanish strings in `community_cue.dart`, `cue_point.dart`, `camelot_constants.dart` | Needs l10n refactor |
| 9 | LOW | `CollectionRule.displayDescription` uses hardcoded English strings | Needs l10n integration |
| 10 | LOW | `energyToPercent` has confusing dual-range handling (0-1 and 1-10) | Works but error-prone |

---

## SECURITY AUDIT

### Strengths
- All SQL queries use parameterized `?` placeholders (no SQL injection)
- HMAC-SHA256 auth for sync endpoints in production
- Input validation on all public endpoints
- Rate limiting configurable per IP
- ETag caching on artwork responses

### Weaknesses Fixed
- CORS now includes required sync headers
- Sync clear endpoint now properly validates admin token
- SQL injection regex no longer blocks legitimate artist names
- Single quotes now allowed in sanitized strings

### Remaining Security Concerns
- `ADMIN_TOKEN` is static with no rotation mechanism
- Rate limiting is per-IP only (not per-user in multi-tenant mode)
- `preview_generator.py` passes file paths to ffmpeg (mitigated by list-based subprocess)
- `allow_credentials=True` with `["*"]` origins in DEBUG mode

---

## ARCHITECTURE NOTES

### Database Connection Pattern
- `AnalysisDB` now uses WAL mode on persistent connection for better concurrency
- Per-query connections (in search methods) are still created without WAL - acceptable for read operations
- `sync_endpoints.py` correctly uses WAL mode on its separate database

### Pydantic Compatibility
- `AnalysisResult` now has `extra="ignore"` to handle cached JSON with extra fields
- `.dict()` is deprecated in Pydantic v2 (should migrate to `.model_dump()` in future)

### Test Coverage
- Backend: 3 test files exist (test_api.py, test_validation.py, test_beatport.py)
- Frontend: No meaningful tests (placeholder only)
- **Recommendation**: Add integration tests for all fixed endpoints

---

# Deep Re-Audit — 2026-04-20

Findings **incremental** respecto al audit original 2026-04-11. Cambios en el
backend desde entonces: admin endpoints, community notes/ratings/popularity,
Redis rate limiter opcional, genre mappings externalizados a JSON,
Pydantic v2 completo, `dict | None → Optional[dict]` para Python 3.9.

Los findings historicos de la tabla "BUGS FIXED" estan verificados como
cerrados. Los de "KNOWN ISSUES NOT FIXED" se revalidan abajo con su estado
actual.

## Nuevos hallazgos HIGH

**B-H1 — `sync_endpoints.py:1099`, `routes/admin.py:42` — timing attack en admin token**
Comparacion con `==` plano en lugar de `hmac.compare_digest()`. Las rutas de
sync HMAC si usan `compare_digest` correctamente (linea 70); la inconsistencia
afecta solo a endpoints admin.
Fix:
```python
import hmac
if not hmac.compare_digest(auth[7:], ADMIN_TOKEN):
    raise HTTPException(403, "invalid admin token")
```

**B-H2 — `main.py:1749` — rate limiting DESACTIVADO en `/analyze`**
`# check_rate_limit(get_client_ip(request))` comentado. El endpoint mas caro
(hasta 100MB, minutos de CPU con librosa) esta expuesto sin limite. Mismo caso
en `/identify` y `/recognize`. En Render Standard single-worker esto es un
vector de DoS trivial. Reactivar con limites estrictos (5-10 req/min/IP para
endpoints de analisis, 60 para los demas).

**B-H3 — `main.py:1752` — header `X-Original-Path` sin validar**
`original_path = request.headers.get("X-Original-Path", "")` se propaga a
`generate_preview_snippet(file_path=original_path, ...)`. Hoy ffmpeg se invoca
como subprocess con lista (no shell=True), asi que no hay RCE, pero es path
traversal latente si algun codepath futuro concatena la ruta a un shell. Validar
con `os.path.abspath()` y whitelist contra ARTWORK_CACHE_DIR / allowed roots.

## Nuevos hallazgos MEDIUM

**B-M1 — `main.py:2034, 2862, 3028` — tres bare `except:`**
Capturan `SystemExit`, `KeyboardInterrupt`, errores de memoria. Enmascaran
bugs y rompen shutdown grace. Cambiar a `except Exception as e: logger.warning(...)`.

**B-M2 — `database.py:204-210+` — cada helper crea conexion nueva**
`AnalysisDB.get_track_by_filename`, `get_analysis`, `save_*`, `update_*` hacen
`sqlite3.connect(self.db_path)` en cada llamada en vez de usar `self.conn`
(que si esta en WAL). Se pierde el beneficio de la conexion persistente y
se duplica overhead en ~10 metodos. Refactor: usar `with self.conn:` para
transacciones, o pool.

**B-M3 — `SELECT *` fragil a column drift**
`database.py:207,215,227,356-416` y `community_cues_endpoint.py:223,257`.
`_row_to_dict` usa indices hardcodeados (0=id, 11=analysis_json, 14=chromaprint).
Cualquier `ALTER TABLE ADD COLUMN` intermedio rompe el mapping. Refactor a
`cursor.description` + dict lookup o Pydantic row models.

**B-M4 — sync endpoints sin rate limiting post-auth**
HMAC valida, pero una vez autenticado un device puede spam `/sync/push` sin
limites. MAX_DEVICES_PER_USER=20 previene creacion masiva, no exfiltracion.
Anadir limite por `device_id` (100 push/min).

**B-M5 — CORS `["*"]` + `allow_credentials=True` en DEBUG**
`main.py:327-328`. Los browsers modernos lo rechazan (RFC 6750). Si `DEBUG=true`
llega a produccion por error, los requests cross-origin fallan silencio.
Cambiar default DEBUG a `["http://localhost:3000", "http://localhost:8000"]`.

**B-M6 — `community_cues_endpoint.py:223,257` — devuelve tuples sin validar**
Queries retornan tuples crudos, response sin Pydantic. Si cambia el schema,
unpack falla. Mapear a `CommunityResponse` model.

## Nuevos hallazgos LOW

**B-L1 — 151 ocurrencias de `print()` en `main.py`**
Polutan stdout de Render, bypass del nivel de logger. Migrar a `logger.info/debug`.

**B-L2 — `config.py:162` no falla al arranque si `ADMIN_TOKEN` vacio en prod**
Print warning y sigue. En Render sin ADMIN_TOKEN, todos los admin endpoints
devuelven 500 opaco. Fail-fast: `if os.environ.get("RENDER") and not ADMIN_TOKEN: sys.exit(1)`.

**B-L3 — `/search/artist/{artist}`, `/search/genre/{genre}` — sin validate**
`main.py:2896-2910` no pasan por `validation.sanitize_string()`. Los otros
`/search/*` si. Anadir validacion por consistencia.

**B-L4 — `preview_generator.py:57` — timeout 15s**
En Render Standard con CPU compartida, archivos de 200+MB pueden fallar por
timeout. Subir a 30-45s via env var `PREVIEW_TIMEOUT_SECONDS`.

**B-L5 — `railway.toml` obsoleto vs CLAUDE.md**
El proyecto despliega en Render (per CLAUDE.md). `railway.toml` sigue en repo
con comentarios referenciando Railway. `.env.example` tambien menciona
`tu-app.railway.app`. O se marca Railway como deploy alternativo (mantener) o
se elimina (mas limpio).

## Verificacion de "KNOWN ISSUES NOT FIXED" del audit 2026-04-11

| # | Issue original | Estado 2026-04-20 |
|---|----------------|-------------------|
| Backend 1 | `has_heavy_bass` usa amplitud inicial | **Sin cambios** — sigue igual |
| Backend 2 | Beatport `track_type_source` sobrescrito | **Sin cambios** |
| Backend 3 | CORS wildcard+credentials en DEBUG | **Sin cambios** (documentado aqui B-M5) |
| Backend 4 | Genero overlap BPM 120-130 | **Sin cambios** (spectral_genre_classifier.py) |
| Backend 5 | `artwork_and_cuepoints.py` hardcodea `/data/artwork_cache` | **CORREGIDO** (commit 286cb49) |
| Backend 6 | `search_analyzed_endpoint.py`, `endpoints_adicionales.py` dead | **ELIMINADOS** (commits 6320a5b, 8de1c10) |
| Backend 7 | `test_beatport.py` no-pytest | **Sin cambios** |

## Resumen por categoria (nuevos hallazgos)

| Categoria | Count |
|-----------|-------|
| Security | 3 (HIGH) + 1 (MED) |
| Database | 2 (MED) |
| API Design | 2 (MED) + 1 (LOW) |
| Logging/Config | 1 (LOW) |
| Dead/obsolete config | 1 (LOW) |

**Total 2026-04-20:** 3 HIGH + 6 MED + 5 LOW = **14 nuevos findings**.
Ninguno bloquea el servicio actual, pero **B-H2 (rate limiting off)** es el mas
urgente — un atacante con curl y un MP3 grande puede saturar la CPU del worker
Render.

