"""
Test de la migracion que arregla el 500 en POST /community/beat-grid
(produccion 2026-07-04).

BUG: `submit_beat_grid_correction` hace `INSERT ... ON CONFLICT(fingerprint,
device_id) DO UPDATE`. Eso EXIGE un UNIQUE/PRIMARY KEY sobre esas columnas. Las
BDs creadas antes de que el CREATE llevara el `UNIQUE(fingerprint, device_id)`
inline NO lo tienen (CREATE TABLE IF NOT EXISTS no altera tablas existentes) ->
sqlite3.OperationalError -> 500 en cada POST.

FIX: init_db crea un UNIQUE INDEX en caliente (idempotente), con dedup previo
por si la tabla vieja acumulo (fingerprint, device_id) repetidos.

Solo depende de sqlite3 (no librosa) -> corre en cualquier entorno.
"""

import os
import sqlite3
import tempfile

from database import AnalysisDB


def _make_old_schema_db():
    """Crea una BD con beat_grid_corrections al estilo VIEJO: sin
    UNIQUE(fingerprint, device_id) y con una fila duplicada."""
    fd, path = tempfile.mkstemp(suffix='.db', prefix='bg_mig_')
    os.close(fd)
    raw = sqlite3.connect(path)
    raw.execute('''
        CREATE TABLE beat_grid_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT NOT NULL,
            device_id TEXT NOT NULL,
            bpm_adjust REAL DEFAULT 0.0,
            beat_offset REAL DEFAULT 0.0,
            original_bpm REAL DEFAULT 0.0
        )
    ''')
    # Duplicado (fp1, devA) — imposible con la constraint, posible sin ella.
    raw.execute("INSERT INTO beat_grid_corrections (fingerprint, device_id, bpm_adjust) VALUES ('fp1','devA',0.1)")
    raw.execute("INSERT INTO beat_grid_corrections (fingerprint, device_id, bpm_adjust) VALUES ('fp1','devA',0.2)")
    raw.commit()
    raw.close()
    return path


def test_old_schema_beat_grid_no_longer_500s():
    path = _make_old_schema_db()
    try:
        # Instanciar AnalysisDB corre init_db -> migracion (dedup + unique index).
        db = AnalysisDB(db_path=path)

        # Antes del fix, esto lanzaba OperationalError (ON CONFLICT sin constraint).
        db.submit_beat_grid_correction(
            fingerprint='fp1', device_id='devA',
            bpm_adjust=0.5, beat_offset=0.01, original_bpm=128.0,
        )
        # Segundo POST del mismo device -> rama UPDATE del ON CONFLICT.
        db.submit_beat_grid_correction(
            fingerprint='fp1', device_id='devA',
            bpm_adjust=0.7, beat_offset=0.02, original_bpm=128.0,
        )

        check = sqlite3.connect(path)
        rows = check.execute(
            "SELECT bpm_adjust FROM beat_grid_corrections "
            "WHERE fingerprint='fp1' AND device_id='devA'"
        ).fetchall()
        check.close()

        # Una sola fila (dedup + upsert), con el ultimo valor.
        assert len(rows) == 1, f"esperaba 1 fila, hay {len(rows)}"
        assert rows[0][0] == 0.7
    finally:
        os.remove(path)


def test_second_device_same_fingerprint_coexists():
    path = _make_old_schema_db()
    try:
        db = AnalysisDB(db_path=path)
        db.submit_beat_grid_correction(
            fingerprint='fp1', device_id='devA',
            bpm_adjust=0.5, beat_offset=0.0, original_bpm=128.0,
        )
        db.submit_beat_grid_correction(
            fingerprint='fp1', device_id='devB',
            bpm_adjust=0.9, beat_offset=0.0, original_bpm=128.0,
        )
        result = db.get_community_beat_grid('fp1')
        # 2 devices distintos para el mismo fingerprint -> 2 contribuidores.
        assert result['contributors'] == 2
        assert result['validated'] is True
    finally:
        os.remove(path)
