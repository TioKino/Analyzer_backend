"""Migrate legacy MD5 fingerprints to Chromaprint.

Script one-shot que recorre la BD `tracks` buscando registros con
`fingerprint_source = 'md5_legacy'` y, si el `file_path` esta accesible
desde el host, recalcula el fingerprint via Chromaprint.

Diseno:
- Idempotente: si lo corres dos veces, la segunda no hace nada.
- Tolerante a fallos: cada track se procesa en su propia transaccion.
- Dry-run por defecto: no escribe nada hasta que se pasa --apply.
- Solo procesa tracks con `file_path` valido. En Render practicamente
  ninguno tendra path local accesible (los uploads van a /tmp y se
  borran). El script esta pensado para LOCAL_ENGINE de developers que
  tienen los archivos originales.

Cambios al actualizar un track:
1. UPDATE tracks SET fingerprint=<nuevo>, id=<nuevo>, fingerprint_source='chromaprint'.
2. UPDATE corrections SET fingerprint=<nuevo>, track_id=<nuevo>.
3. UPDATE community_cues SET fingerprint=<nuevo>.
4. UPDATE community_notes SET fingerprint=<nuevo>.
5. UPDATE beat_grid_corrections SET fingerprint=<nuevo>.
6. UPDATE track_popularity SET fingerprint=<nuevo>.
7. UPDATE track_ratings SET fingerprint=<nuevo>.
8. UPDATE audd_call_log SET fingerprint=<nuevo>.
9. UPDATE dj_notes SET track_id=<nuevo>, fingerprint=<nuevo>.
10. Renombrar previews_cache/<old>.mp3 a previews_cache/<nuevo>.mp3 si existe.

Uso:
    python scripts/migrate_to_chromaprint.py --db /data/analysis.db
    python scripts/migrate_to_chromaprint.py --db /data/analysis.db --apply
    python scripts/migrate_to_chromaprint.py --db local_analysis.db --apply --previews-dir ./previews_cache

NO ejecutar en production directamente sin un dry-run previo y backup.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
from typing import Optional

# Permite ejecutar desde la raiz del repo: `python scripts/migrate_to_chromaprint.py ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromaprint_helper import (  # noqa: E402
    ChromaprintFailed,
    ChromaprintUnavailable,
    calculate_chromaprint_fingerprint,
)

logger = logging.getLogger("migrate_to_chromaprint")


# Tablas que usan `fingerprint` como columna y deben actualizarse en cascada.
# `dj_notes` y `corrections` tambien tienen `track_id` (que era == fingerprint).
FINGERPRINT_TABLES = (
    ("corrections", ("fingerprint", "track_id")),
    ("dj_notes", ("fingerprint", "track_id")),
    ("community_cues", ("fingerprint",)),
    ("community_notes", ("fingerprint",)),
    ("beat_grid_corrections", ("fingerprint",)),
    ("track_popularity", ("fingerprint",)),
    ("track_ratings", ("fingerprint",)),
    ("audd_call_log", ("fingerprint",)),
)


def find_track_path(track: sqlite3.Row, search_dirs: list[str]) -> Optional[str]:
    """Devuelve la ruta absoluta del archivo de audio del track, o None.

    Estrategia:
    1. Si `analysis_json.file_path` existe y el archivo esta en disco, lo usa.
    2. Si no, busca por filename en cada dir de `search_dirs`.
    """
    import json as _json

    # 1. Intentar file_path embebido en analysis_json.
    aj = track["analysis_json"]
    if aj:
        try:
            data = _json.loads(aj)
            fp = data.get("file_path")
            if fp and os.path.isfile(fp):
                return fp
        except (_json.JSONDecodeError, TypeError):
            pass

    # 2. Buscar por filename en search_dirs.
    filename = track["filename"]
    if filename and search_dirs:
        for d in search_dirs:
            candidate = os.path.join(d, filename)
            if os.path.isfile(candidate):
                return candidate

    return None


def migrate_track(
    conn: sqlite3.Connection,
    old_fingerprint: str,
    old_id: str,
    audio_path: str,
    previews_dir: Optional[str],
    apply_changes: bool,
) -> tuple[bool, Optional[str]]:
    """Migra un track. Devuelve (ok, new_fingerprint_or_None).

    Si `apply_changes=False` (dry-run), calcula el fingerprint y reporta
    pero no escribe en BD ni mueve archivos.
    """
    try:
        new_fp, _b64, _dur_ms = calculate_chromaprint_fingerprint(audio_path)
    except (ChromaprintUnavailable, ChromaprintFailed) as exc:
        logger.warning(
            "Chromaprint fallo en %s (id=%s): %s",
            audio_path, old_id[:12], exc,
        )
        return False, None

    if new_fp == old_fingerprint:
        # Caso raro pero posible: el old_fingerprint ya era Chromaprint
        # marcado erroneamente como md5_legacy. Lo arreglamos sin renombrar.
        if apply_changes:
            conn.execute(
                "UPDATE tracks SET fingerprint_source='chromaprint' WHERE id=?",
                (old_id,),
            )
            conn.commit()
        logger.info(
            "Track %s ya tenia fingerprint Chromaprint correcto, solo marca de origen actualizada",
            old_id[:12],
        )
        return True, new_fp

    if not apply_changes:
        logger.info(
            "[DRY-RUN] Migraria %s -> %s (file=%s)",
            old_id[:12], new_fp[:12], os.path.basename(audio_path),
        )
        return True, new_fp

    # Comprobar colision: si new_fp ya existe como otro track en BD,
    # NO sobreescribimos. Loguemos y dejamos el track como md5_legacy.
    existing = conn.execute(
        "SELECT id FROM tracks WHERE id = ? OR fingerprint = ?",
        (new_fp, new_fp),
    ).fetchone()
    if existing and existing["id"] != old_id:
        logger.warning(
            "Colision: new_fp=%s ya existe como track %s. Saltando %s.",
            new_fp[:12], existing["id"][:12], old_id[:12],
        )
        return False, None

    # Actualizar la fila principal y todas las tablas relacionadas atomicamente.
    try:
        conn.execute(
            "UPDATE tracks SET fingerprint=?, id=?, fingerprint_source='chromaprint' WHERE id=?",
            (new_fp, new_fp, old_id),
        )
        for table, cols in FINGERPRINT_TABLES:
            for col in cols:
                conn.execute(
                    f"UPDATE {table} SET {col}=? WHERE {col}=?",
                    (new_fp, old_id if col == "track_id" else old_fingerprint),
                )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.error("UPDATE fallo para %s: %s", old_id[:12], exc)
        return False, None

    # Renombrar preview en disco.
    if previews_dir:
        old_preview = os.path.join(previews_dir, f"{old_fingerprint}.mp3")
        new_preview = os.path.join(previews_dir, f"{new_fp}.mp3")
        if os.path.isfile(old_preview):
            try:
                shutil.move(old_preview, new_preview)
                logger.debug("Preview renombrado: %s -> %s", old_preview, new_preview)
            except OSError as exc:
                logger.warning("No pude renombrar preview %s: %s", old_preview, exc)

    logger.info(
        "Migrado %s -> %s (file=%s)",
        old_id[:12], new_fp[:12], os.path.basename(audio_path),
    )
    return True, new_fp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--db",
        default=os.getenv("DATABASE_PATH", "/data/analysis.db"),
        help="Ruta al sqlite (default: $DATABASE_PATH o /data/analysis.db)",
    )
    parser.add_argument(
        "--previews-dir",
        default=os.getenv("PREVIEWS_DIR", "/data/previews_cache"),
        help="Directorio de previews para renombrar archivos (.mp3)",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Directorios extra para buscar el archivo de audio por filename. "
             "Repetible: --search-dir /path1 --search-dir /path2",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica cambios. Sin --apply solo hace dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Procesar como mucho N tracks (0 = todos).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.isfile(args.db):
        logger.error("BD no encontrada: %s", args.db)
        return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Asegurar columna fingerprint_source (idempotente).
    try:
        conn.execute(
            "ALTER TABLE tracks ADD COLUMN fingerprint_source TEXT DEFAULT 'md5_legacy'"
        )
        conn.commit()
        logger.info("Columna fingerprint_source anadida (era BD pre-v2.8.0).")
    except sqlite3.OperationalError:
        pass  # Ya existe

    query = """
        SELECT id, filename, fingerprint, fingerprint_source, analysis_json
        FROM tracks
        WHERE fingerprint_source = 'md5_legacy' OR fingerprint_source IS NULL
        ORDER BY analyzed_at DESC
    """
    if args.limit:
        query += f" LIMIT {int(args.limit)}"

    rows = conn.execute(query).fetchall()
    logger.info("Tracks legacy en BD: %d", len(rows))

    stats = {"migrated": 0, "skipped_no_path": 0, "skipped_failed": 0}

    for row in rows:
        old_id = row["id"]
        old_fingerprint = row["fingerprint"] or old_id
        audio_path = find_track_path(row, args.search_dir)
        if not audio_path:
            stats["skipped_no_path"] += 1
            logger.debug(
                "Skip %s: archivo no accesible (filename=%s)",
                old_id[:12], row["filename"],
            )
            continue

        ok, _new_fp = migrate_track(
            conn=conn,
            old_fingerprint=old_fingerprint,
            old_id=old_id,
            audio_path=audio_path,
            previews_dir=args.previews_dir if os.path.isdir(args.previews_dir or "") else None,
            apply_changes=args.apply,
        )
        if ok:
            stats["migrated"] += 1
        else:
            stats["skipped_failed"] += 1

    conn.close()

    mode = "APLICADO" if args.apply else "DRY-RUN (usa --apply para escribir)"
    logger.info(
        "Resumen [%s]: migrados=%d, sin_path=%d, fallidos=%d, total_legacy=%d",
        mode,
        stats["migrated"],
        stats["skipped_no_path"],
        stats["skipped_failed"],
        len(rows),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
