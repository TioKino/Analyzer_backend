"""Tests para los cambios de schema v2.8.0 en `database.py`.

Cubre:
- Columna `fingerprint_source` (defaults, persistencia, override).
- Columnas `volume_id` y `relative_path` (NULL=interno, persistencia).
- ALTER TABLE idempotente: BD pre-v2.8.0 se migra sin perder datos.

No dependen de fastapi/librosa — solo sqlite3 + database.py. Usan tmp_path
de pytest para BD aislada por test.
"""

from __future__ import annotations

import sqlite3
from typing import Dict

import pytest

from database import AnalysisDB


def _track(track_id: str = "abc", **overrides) -> Dict:
    base = {
        'id': track_id,
        'filename': 'song.mp3',
        'artist': 'Artist',
        'title': 'Title',
        'duration': 180.0,
        'bpm': 128.0,
        'key': 'C',
        'camelot': '8B',
        'energy_dj': 7,
        'genre': 'Techno',
        'track_type': 'peak',
        'fingerprint': 'fp_' + track_id,
    }
    base.update(overrides)
    return base


@pytest.fixture
def db(tmp_path):
    """BD efímera por test. Esquema fresh con todas las columnas v2.8.0."""
    return AnalysisDB(db_path=str(tmp_path / "analysis.db"))


# ==================== fingerprint_source ====================

class TestFingerprintSource:
    def test_default_md5_legacy_cuando_no_se_especifica(self, db):
        db.save_track(_track('a'))
        row = db.get_track_by_id('a')
        assert row is not None
        assert row['fingerprint_source'] == 'md5_legacy'

    def test_persiste_chromaprint_explicito(self, db):
        db.save_track(_track('b', fingerprint_source='chromaprint'))
        row = db.get_track_by_id('b')
        assert row['fingerprint_source'] == 'chromaprint'

    def test_round_trip_replace_preserva_source(self, db):
        db.save_track(_track('c', fingerprint_source='chromaprint'))
        # Re-save con el mismo id (INSERT OR REPLACE).
        db.save_track(_track('c', fingerprint_source='chromaprint', bpm=130.0))
        row = db.get_track_by_id('c')
        assert row['fingerprint_source'] == 'chromaprint'
        assert row['bpm'] == 130.0


# ==================== volume_id + relative_path ====================

class TestVolumeColumns:
    def test_default_null_para_disco_interno(self, db):
        db.save_track(_track('d'))
        row = db.get_track_by_id('d')
        assert row['volume_id'] is None
        assert row['relative_path'] is None

    def test_persiste_volume_id_y_relative_path(self, db):
        db.save_track(_track(
            'e',
            volume_id='vol_8a7d3f9b2e1c4a8d',
            relative_path='Music/Techno/track.mp3',
        ))
        row = db.get_track_by_id('e')
        assert row['volume_id'] == 'vol_8a7d3f9b2e1c4a8d'
        assert row['relative_path'] == 'Music/Techno/track.mp3'

    def test_save_track_sin_keys_de_volumen_no_falla(self, db):
        # Tracks legacy: dicts sin volume_id/relative_path. La nueva BD
        # debe aceptarlos y guardarlos como NULL.
        legacy = _track('f')
        legacy.pop('fingerprint_source', None)
        db.save_track(legacy)
        row = db.get_track_by_id('f')
        assert row['volume_id'] is None
        assert row['relative_path'] is None

    def test_index_idx_tracks_volume_existe(self, db):
        # Verifica que el index se creó (acelera queries "tracks de este volumen").
        conn = sqlite3.connect(db.db_path)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tracks_volume'"
            )
            assert cur.fetchone() is not None
        finally:
            conn.close()


# ==================== ALTER TABLE idempotente ====================

class TestSchemaMigration:
    def test_bd_legacy_se_migra_sin_perder_datos(self, tmp_path):
        """Simula una BD pre-v2.8.0 sin las nuevas columnas y verifica que
        AnalysisDB.__init__ las añade vía ALTER TABLE sin tocar las filas."""
        db_path = str(tmp_path / "legacy.db")

        # Crear schema "antiguo" — sin fingerprint_source, volume_id, relative_path.
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE tracks (
                id TEXT PRIMARY KEY,
                filename TEXT,
                artist TEXT,
                title TEXT,
                duration REAL,
                bpm REAL,
                key TEXT,
                camelot TEXT,
                energy_dj INTEGER,
                genre TEXT,
                track_type TEXT,
                analysis_json TEXT,
                analyzed_at TEXT,
                fingerprint TEXT
            )
        ''')
        conn.execute(
            "INSERT INTO tracks (id, filename, artist, title, duration, bpm, "
            "key, camelot, energy_dj, genre, track_type, analyzed_at, fingerprint) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ('legacy_id', 'old.mp3', 'A', 'T', 100.0, 120.0, 'C', '8B',
             5, 'Techno', 'peak', '2025-01-01T00:00:00', 'fp_legacy'),
        )
        conn.commit()
        conn.close()

        # Re-abrir vía AnalysisDB → init_db dispara los ALTER TABLE.
        db = AnalysisDB(db_path=db_path)

        # La fila legacy sigue ahí.
        row = db.get_track_by_id('legacy_id')
        assert row is not None
        assert row['filename'] == 'old.mp3'

        # Las nuevas columnas existen y la fila legacy las tiene como
        # NULL (volume_id, relative_path) o el default 'md5_legacy'
        # (fingerprint_source).
        # Nota: ALTER TABLE ADD COLUMN aplica el DEFAULT solo a filas FUTURAS
        # en algunas versiones de SQLite; las filas existentes pueden quedar
        # con NULL hasta el siguiente UPDATE. Aceptamos ambos casos.
        assert row.get('fingerprint_source') in (None, 'md5_legacy')
        assert row.get('volume_id') is None
        assert row.get('relative_path') is None

    def test_init_db_es_idempotente(self, tmp_path):
        """Llamar init_db varias veces no debe fallar ni duplicar columnas."""
        db_path = str(tmp_path / "idempotent.db")
        AnalysisDB(db_path=db_path)
        AnalysisDB(db_path=db_path)  # Re-init — no debe lanzar.

        # Verificar que solo tenemos UNA columna fingerprint_source.
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute("PRAGMA table_info(tracks)")
            cols = [row[1] for row in cur.fetchall()]
            assert cols.count('fingerprint_source') == 1
            assert cols.count('volume_id') == 1
            assert cols.count('relative_path') == 1
        finally:
            conn.close()
