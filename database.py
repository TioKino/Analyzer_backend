import sqlite3
import os
import time
from datetime import datetime, timezone
import json
from typing import List, Dict, Optional

class AnalysisDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "/data/analysis.db")
        self.db_path = db_path
        self._conn = None  # Conexion persistente
        self.init_db()

    def _open_conn(self):
        """Abre conexion nueva con row_factory = sqlite3.Row.

        Habilita acceso a columnas por NOMBRE (row['artist'], row.keys())
        ademas del acceso por indice tradicional (row[2]). El nuevo
        acceso por nombre es robusto frente a ALTER TABLE ADD COLUMN
        y elimina la fragilidad que documentaba B-M3 del AUDIT 2026-04-20
        (indices hardcodeados en _row_to_dict).

        Las funciones existentes que usan row[N] siguen funcionando
        sin cambios porque sqlite3.Row soporta ambas formas de acceso.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def conn(self):
        """Conexion persistente con WAL + row_factory (lazy)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_db(self):
        # En init_db no consultamos columnas, solo creamos tablas/indices,
        # asi que no hace falta row_factory.
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Tabla de analisis
        c.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
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
                fingerprint TEXT,
                chromaprint TEXT
            )
        ''')

        # Migracion: anadir columna chromaprint si no existe (BDs antiguas)
        try:
            c.execute('ALTER TABLE tracks ADD COLUMN chromaprint TEXT')
        except sqlite3.OperationalError:
            pass  # Ya existe

        # Indices para busquedas rapidas
        c.execute('CREATE INDEX IF NOT EXISTS idx_artist ON tracks(artist)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_genre ON tracks(genre)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_bpm ON tracks(bpm)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_energy ON tracks(energy_dj)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_key ON tracks(key)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_camelot ON tracks(camelot)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_chromaprint ON tracks(chromaprint)')

        # Correcciones manuales (memoria colectiva)
        c.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT,
                field TEXT,
                old_value TEXT,
                new_value TEXT,
                corrected_at TEXT,
                fingerprint TEXT,
                FOREIGN KEY (track_id) REFERENCES tracks(id)
            )
        ''')

        # Notas DJ
        c.execute('''
            CREATE TABLE IF NOT EXISTS dj_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id TEXT,
                fingerprint TEXT,
                note TEXT,
                created_at TEXT,
                FOREIGN KEY (track_id) REFERENCES tracks(id)
            )
        ''')

        # Community cues (community_cues_endpoint.py)
        c.execute('''
            CREATE TABLE IF NOT EXISTS community_cues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                device_id TEXT NOT NULL,
                cue_type TEXT NOT NULL,
                position_ms INTEGER NOT NULL,
                end_position_ms INTEGER,
                note TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(fingerprint, device_id, cue_type, position_ms)
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_cc_fingerprint ON community_cues(fingerprint)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_cc_device ON community_cues(device_id)')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS beat_grid_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                device_id TEXT NOT NULL,
                bpm_adjust REAL DEFAULT 0.0,
                beat_offset REAL DEFAULT 0.0,
                original_bpm REAL DEFAULT 0.0,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(fingerprint, device_id)
            )
        ''')

        # Notas comunitarias (todos los DJs ven las notas de todos)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS community_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                device_id TEXT NOT NULL,
                display_name TEXT DEFAULT 'DJ',
                note_text TEXT NOT NULL,
                note_type TEXT DEFAULT 'general',
                upvotes INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(fingerprint, device_id, note_text)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cnotes_fp ON community_notes(fingerprint)')

        # Popularidad de tracks
        conn.execute('''
            CREATE TABLE IF NOT EXISTS track_popularity (
                fingerprint TEXT PRIMARY KEY,
                analysis_count INTEGER DEFAULT 1,
                dj_count INTEGER DEFAULT 1,
                avg_rating REAL DEFAULT 0,
                total_ratings INTEGER DEFAULT 0,
                last_analyzed TEXT
            )
        ''')

        # Ratings individuales por DJ
        conn.execute('''
            CREATE TABLE IF NOT EXISTS track_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                device_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                rated_at TEXT NOT NULL,
                UNIQUE(fingerprint, device_id)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_tr_fp ON track_ratings(fingerprint)')

        # AudD auto-trigger: log de llamadas para honrar cooldown y daily cap.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audd_call_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                called_at REAL NOT NULL,
                success INTEGER NOT NULL,
                artist TEXT,
                title TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_audd_fp ON audd_call_log(fingerprint)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_audd_at ON audd_call_log(called_at)')

        conn.commit()
        conn.close()

    def _row_to_dict(self, row) -> Optional[Dict]:
        """Convierte una sqlite3.Row de la tabla `tracks` a dict.

        Antes (pre B-M3): indices hardcodeados (row[0]=id, row[11]=analysis_json,
        row[14]=chromaprint). Cualquier ALTER TABLE intermedio rompia el
        mapping silenciosamente. Ahora iteramos por `row.keys()`, asi que
        si manana se anade una columna se incluye automaticamente sin
        tocar este metodo.
        """
        if not row:
            return None

        # sqlite3.Row.keys() devuelve los nombres de las columnas del SELECT.
        base_dict = {k: row[k] for k in row.keys()}

        # Enriquecer con campos guardados dentro de analysis_json (artwork_url,
        # label, year, etc.) que no estan como columnas propias.
        analysis_json = base_dict.get('analysis_json')
        if analysis_json:
            try:
                full_analysis = json.loads(analysis_json)
                for key in ['artwork_url', 'artwork_embedded', 'label', 'year', 'album',
                           'isrc', 'bpm_source', 'key_source', 'genre_source',
                           'cue_points', 'first_beat', 'beat_interval', 'drop_timestamp']:
                    if key in full_analysis and full_analysis[key] is not None:
                        base_dict[key] = full_analysis[key]
            except (json.JSONDecodeError, TypeError):
                pass

        return base_dict

    def _rows_to_list(self, rows) -> List[Dict]:
        return [self._row_to_dict(row) for row in rows if row]

    def get_track_by_filename(self, filename):
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM tracks WHERE filename = ?', (filename,))
        result = c.fetchone()
        conn.close()
        return result

    def get_track_by_id(self, track_id: str) -> Optional[Dict]:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM tracks WHERE id = ?', (track_id,))
        result = c.fetchone()
        conn.close()
        return self._row_to_dict(result)

    def get_track_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        """Busca un track por su fingerprint de audio (memoria colectiva)"""
        if not fingerprint:
            return None
        conn = self._open_conn()
        c = conn.cursor()
        # Buscar por fingerprint O por id (que a veces es el fingerprint)
        c.execute('SELECT * FROM tracks WHERE fingerprint = ? OR id = ?', (fingerprint, fingerprint))
        result = c.fetchone()
        conn.close()
        return self._row_to_dict(result)

    def save_track(self, track_data):
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            INSERT OR REPLACE INTO tracks
            (id, filename, artist, title, duration, bpm, key, camelot,
             energy_dj, genre, track_type, analysis_json, analyzed_at, fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            track_data['id'],
            track_data['filename'],
            track_data.get('artist'),
            track_data.get('title'),
            track_data['duration'],
            track_data['bpm'],
            track_data.get('key'),
            track_data.get('camelot'),
            track_data['energy_dj'],
            track_data['genre'],
            track_data['track_type'],
            json.dumps(track_data),
            datetime.now().isoformat(),
            track_data.get('fingerprint')
        ))

        conn.commit()
        conn.close()

    def save_correction(self, track_id, field, old_value, new_value, fingerprint=None, device_id=None):
        conn = self._open_conn()
        c = conn.cursor()

        if device_id and fingerprint:
            c.execute('''
                DELETE FROM corrections
                WHERE fingerprint = ? AND field = ? AND track_id = ?
                AND corrected_at IN (
                    SELECT corrected_at FROM corrections
                    WHERE fingerprint = ? AND field = ? AND track_id = ?
                    ORDER BY corrected_at DESC LIMIT 1
                )
            ''', (fingerprint, field, track_id, fingerprint, field, track_id))

        c.execute('''
            INSERT INTO corrections
            (track_id, field, old_value, new_value, corrected_at, fingerprint)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (track_id, field, old_value, new_value, datetime.now().isoformat(), fingerprint))

        conn.commit()
        conn.close()

    def get_consensus(self, fingerprint, field, min_votes=1):
        """
        Obtiene el valor con mas votos para un campo de un track.
        Devuelve (value, vote_count) o (None, 0) si no hay consenso suficiente.
        """
        if not fingerprint:
            return None, 0

        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT new_value, COUNT(DISTINCT track_id || corrected_at) as vote_count
            FROM corrections
            WHERE fingerprint = ? AND field = ?
            GROUP BY new_value
            ORDER BY vote_count DESC
            LIMIT 1
        ''', (fingerprint, field))

        result = c.fetchone()
        conn.close()

        if result and result['vote_count'] >= min_votes:
            return result['new_value'], result['vote_count']
        return None, 0

    def get_collective_genre(self, fingerprint):
        """Legacy: usa get_consensus con minimo 3 votos."""
        value, count = self.get_consensus(fingerprint, 'genre', min_votes=3)
        return value

    def get_all_consensus(self, fingerprint):
        """
        Obtiene consenso para todos los campos de un track.
        Devuelve dict con {field: (value, vote_count)} para campos con votos.
        """
        if not fingerprint:
            return {}

        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT field, new_value, COUNT(DISTINCT track_id || corrected_at) as vote_count
            FROM corrections
            WHERE fingerprint = ?
            GROUP BY field, new_value
            ORDER BY field, vote_count DESC
        ''', (fingerprint,))

        rows = c.fetchall()
        conn.close()

        result = {}
        for row in rows:
            field = row['field']
            value = row['new_value']
            count = row['vote_count']
            if field not in result or count > result[field][1]:
                result[field] = (value, count)

        return result

    # ==================== BUSQUEDAS ====================

    def search_by_artist_title(self, artist: str, title: str):
        conn = self._open_conn()
        c = conn.cursor()

        artist_pattern = f"%{artist.lower()}%"
        title_pattern = f"%{title.lower()}%"

        c.execute('''
            SELECT * FROM tracks
            WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
            ORDER BY analyzed_at DESC
            LIMIT 10
        ''', (artist_pattern, title_pattern))

        results = c.fetchall()
        conn.close()
        return results

    def search_by_artist(self, artist: str, limit: int = 50) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT * FROM tracks
            WHERE artist LIKE ? COLLATE NOCASE
            ORDER BY artist, title
            LIMIT ?
        ''', (f'%{artist}%', limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def search_by_genre(self, genre: str, limit: int = 100) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT * FROM tracks
            WHERE genre LIKE ? COLLATE NOCASE
            ORDER BY energy_dj DESC, bpm
            LIMIT ?
        ''', (f'%{genre}%', limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def search_by_bpm_range(self, min_bpm: float, max_bpm: float, limit: int = 100) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT * FROM tracks
            WHERE bpm BETWEEN ? AND ?
            ORDER BY bpm
            LIMIT ?
        ''', (min_bpm, max_bpm, limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def search_by_energy(self, min_energy: int, max_energy: int = 10, limit: int = 100) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT * FROM tracks
            WHERE energy_dj BETWEEN ? AND ?
            ORDER BY energy_dj DESC, bpm
            LIMIT ?
        ''', (min_energy, max_energy, limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def search_by_key(self, key: str, limit: int = 100) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT * FROM tracks
            WHERE key = ? COLLATE NOCASE OR camelot = ? COLLATE NOCASE
            ORDER BY bpm, energy_dj
            LIMIT ?
        ''', (key, key, limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def search_compatible_keys(self, camelot: str, limit: int = 50) -> List[Dict]:
        if not camelot or len(camelot) < 2:
            return []

        try:
            number = int(camelot[:-1])
            letter = camelot[-1].upper()
        except (ValueError, IndexError):
            return []

        compatible = [camelot]
        prev_num = 12 if number == 1 else number - 1
        next_num = 1 if number == 12 else number + 1
        compatible.append(f'{prev_num}{letter}')
        compatible.append(f'{next_num}{letter}')
        other_letter = 'B' if letter == 'A' else 'A'
        compatible.append(f'{number}{other_letter}')

        conn = self._open_conn()
        c = conn.cursor()

        placeholders = ','.join(['?' for _ in compatible])
        c.execute(f'''
            SELECT * FROM tracks
            WHERE camelot IN ({placeholders})
            ORDER BY CASE camelot WHEN ? THEN 0 ELSE 1 END, energy_dj DESC
            LIMIT ?
        ''', (*compatible, camelot, limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def search_by_track_type(self, track_type: str, limit: int = 100) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('''
            SELECT * FROM tracks
            WHERE track_type LIKE ? COLLATE NOCASE
            ORDER BY energy_dj, bpm
            LIMIT ?
        ''', (f'%{track_type}%', limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    # Columnas validas para busquedas - whitelist de seguridad
    _VALID_COLUMNS = frozenset({
        'artist', 'genre', 'bpm', 'energy_dj', 'key', 'camelot', 'track_type',
        'title', 'filename', 'duration', 'fingerprint'
    })

    def search_advanced(self, artist=None, genre=None, min_bpm=None, max_bpm=None,
                       min_energy=None, max_energy=None, key=None, track_type=None, limit=100):
        """Busqueda avanzada con multiples filtros.
        SEGURIDAD: Las condiciones WHERE usan columnas hardcoded (no de user input).
        Los valores siempre van por parametros '?' (nunca interpolados)."""
        conn = self._open_conn()
        c = conn.cursor()

        conditions = []
        params = []

        if artist:
            conditions.append('artist LIKE ? COLLATE NOCASE')
            params.append(f'%{artist}%')
        if genre:
            conditions.append('genre LIKE ? COLLATE NOCASE')
            params.append(f'%{genre}%')
        if min_bpm is not None:
            conditions.append('bpm >= ?')
            params.append(min_bpm)
        if max_bpm is not None:
            conditions.append('bpm <= ?')
            params.append(max_bpm)
        if min_energy is not None:
            conditions.append('energy_dj >= ?')
            params.append(min_energy)
        if max_energy is not None:
            conditions.append('energy_dj <= ?')
            params.append(max_energy)
        if key:
            conditions.append('(key = ? COLLATE NOCASE OR camelot = ? COLLATE NOCASE)')
            params.extend([key, key])
        if track_type:
            conditions.append('track_type LIKE ? COLLATE NOCASE')
            params.append(f'%{track_type}%')

        where_clause = ' AND '.join(conditions) if conditions else '1=1'

        c.execute(f'''
            SELECT * FROM tracks
            WHERE {where_clause}
            ORDER BY energy_dj DESC, bpm
            LIMIT ?
        ''', (*params, limit))

        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def get_all_tracks(self, limit: int = 1000) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT * FROM tracks ORDER BY analyzed_at DESC LIMIT ?', (limit,))
        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)

    def get_unique_artists(self) -> List[str]:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            SELECT DISTINCT artist FROM tracks
            WHERE artist IS NOT NULL AND artist != ''
            ORDER BY artist COLLATE NOCASE
        ''')
        results = [row['artist'] for row in c.fetchall()]
        conn.close()
        return results

    def get_unique_genres(self) -> List[str]:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            SELECT DISTINCT genre FROM tracks
            WHERE genre IS NOT NULL AND genre != ''
            ORDER BY genre COLLATE NOCASE
        ''')
        results = [row['genre'] for row in c.fetchall()]
        conn.close()
        return results

    def get_stats(self) -> Dict:
        conn = self._open_conn()
        c = conn.cursor()

        c.execute('SELECT COUNT(*) AS n FROM tracks')
        total_tracks = c.fetchone()['n']

        c.execute('SELECT COUNT(DISTINCT artist) AS n FROM tracks WHERE artist IS NOT NULL')
        unique_artists = c.fetchone()['n']

        c.execute('SELECT COUNT(DISTINCT genre) AS n FROM tracks WHERE genre IS NOT NULL')
        unique_genres = c.fetchone()['n']

        c.execute('SELECT AVG(bpm) AS v FROM tracks')
        avg_bpm = c.fetchone()['v'] or 0

        c.execute('SELECT AVG(energy_dj) AS v FROM tracks')
        avg_energy = c.fetchone()['v'] or 0

        c.execute('SELECT SUM(duration) AS v FROM tracks')
        total_duration = c.fetchone()['v'] or 0

        conn.close()

        return {
            'total_tracks': total_tracks,
            'unique_artists': unique_artists,
            'unique_genres': unique_genres,
            'avg_bpm': round(avg_bpm, 1),
            'avg_energy': round(avg_energy, 1),
            'total_duration_hours': round(total_duration / 3600, 1)
        }

    def save_dj_note(self, track_id: str, note: str, fingerprint: Optional[str] = None):
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            INSERT INTO dj_notes (track_id, fingerprint, note, created_at)
            VALUES (?, ?, ?, ?)
        ''', (track_id, fingerprint, note, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_dj_notes(self, track_id: str) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            SELECT note, created_at FROM dj_notes
            WHERE track_id = ? ORDER BY created_at DESC
        ''', (track_id,))
        results = [{'note': row['note'], 'created_at': row['created_at']} for row in c.fetchall()]
        conn.close()
        return results

    def delete_track(self, track_id: str) -> bool:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('DELETE FROM tracks WHERE id = ?', (track_id,))
        deleted = c.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def delete_track_by_filename(self, filename: str) -> bool:
        """Elimina un track por su filename para permitir reanalisis completo"""
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('DELETE FROM tracks WHERE filename = ?', (filename,))
        deleted = c.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    # ==================== COMMUNITY NOTES ====================

    def save_community_note(self, fingerprint: str, device_id: str,
                            note_text: str, display_name: str = 'DJ',
                            note_type: str = 'general') -> int:
        from datetime import datetime
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO community_notes
            (fingerprint, device_id, display_name, note_text, note_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (fingerprint, device_id, display_name, note_text, note_type,
              datetime.utcnow().isoformat()))
        note_id = c.lastrowid
        conn.commit()
        conn.close()
        return note_id

    def get_community_notes(self, fingerprint: str) -> List[Dict]:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            SELECT id, device_id, display_name, note_text, note_type, upvotes, created_at
            FROM community_notes WHERE fingerprint = ?
            ORDER BY upvotes DESC, created_at DESC
        ''', (fingerprint,))
        results = [{
            'id': r['id'],
            'device_id': r['device_id'][:8] + '...',
            'display_name': r['display_name'],
            'note_text': r['note_text'],
            'note_type': r['note_type'],
            'upvotes': r['upvotes'],
            'created_at': r['created_at'],
        } for r in c.fetchall()]
        conn.close()
        return results

    def upvote_community_note(self, note_id: int):
        conn = self._open_conn()
        conn.execute('UPDATE community_notes SET upvotes = upvotes + 1 WHERE id = ?', (note_id,))
        conn.commit()
        conn.close()

    # ==================== TRACK POPULARITY & RATINGS ====================

    def increment_popularity(self, fingerprint: str, device_id: str = ''):
        from datetime import datetime
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT analysis_count FROM track_popularity WHERE fingerprint = ?', (fingerprint,))
        row = c.fetchone()
        now = datetime.utcnow().isoformat()
        if row:
            c.execute('''UPDATE track_popularity
                         SET analysis_count = analysis_count + 1, last_analyzed = ?
                         WHERE fingerprint = ?''', (now, fingerprint))
        else:
            c.execute('''INSERT INTO track_popularity (fingerprint, analysis_count, dj_count, last_analyzed)
                         VALUES (?, 1, 1, ?)''', (fingerprint, now))
        conn.commit()
        conn.close()

    def rate_track(self, fingerprint: str, device_id: str, rating: int) -> Dict:
        from datetime import datetime
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            INSERT INTO track_ratings (fingerprint, device_id, rating, rated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(fingerprint, device_id) DO UPDATE SET rating = ?, rated_at = ?
        ''', (fingerprint, device_id, rating, datetime.utcnow().isoformat(),
              rating, datetime.utcnow().isoformat()))
        # Recalcular media
        c.execute('SELECT AVG(rating) AS avg, COUNT(*) AS cnt FROM track_ratings WHERE fingerprint = ?', (fingerprint,))
        agg = c.fetchone()
        avg = agg['avg']
        count = agg['cnt']
        c.execute('''
            INSERT INTO track_popularity (fingerprint, avg_rating, total_ratings, last_analyzed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET avg_rating = ?, total_ratings = ?
        ''', (fingerprint, avg or 0, count or 0, datetime.utcnow().isoformat(), avg or 0, count or 0))
        conn.commit()
        conn.close()
        return {'avg_rating': round(avg or 0, 1), 'total_ratings': count or 0}

    def get_track_popularity(self, fingerprint: str) -> Dict:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT analysis_count, dj_count, avg_rating, total_ratings FROM track_popularity WHERE fingerprint = ?',
                  (fingerprint,))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                'analysis_count': row['analysis_count'],
                'dj_count': row['dj_count'],
                'avg_rating': round(row['avg_rating'] or 0, 1),
                'total_ratings': row['total_ratings'] or 0,
            }
        return {'analysis_count': 0, 'dj_count': 0, 'avg_rating': 0, 'total_ratings': 0}

    def get_my_rating(self, fingerprint: str, device_id: str) -> int:
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT rating FROM track_ratings WHERE fingerprint = ? AND device_id = ?',
                  (fingerprint, device_id))
        row = c.fetchone()
        conn.close()
        return row['rating'] if row else 0

    # ==================== COMMUNITY BEAT GRID ====================

    def submit_beat_grid_correction(self, fingerprint: str, device_id: str,
                                     bpm_adjust: float, beat_offset: float,
                                     original_bpm: float):
        """Guarda o actualiza la correccion de beat grid de un DJ"""
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            INSERT INTO beat_grid_corrections (fingerprint, device_id, bpm_adjust, beat_offset, original_bpm, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint, device_id) DO UPDATE SET
                bpm_adjust = excluded.bpm_adjust,
                beat_offset = excluded.beat_offset,
                original_bpm = excluded.original_bpm,
                updated_at = excluded.updated_at
        ''', (fingerprint, device_id, bpm_adjust, beat_offset, original_bpm, now, now))
        conn.commit()
        conn.close()

    def get_community_beat_grid(self, fingerprint: str) -> Dict:
        """Obtiene la correccion promedio de la comunidad para un track"""
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('''
            SELECT AVG(bpm_adjust) AS bpm_adj, AVG(beat_offset) AS beat_off,
                   COUNT(*) AS contributors, AVG(original_bpm) AS orig_bpm
            FROM beat_grid_corrections
            WHERE fingerprint = ?
        ''', (fingerprint,))
        row = c.fetchone()
        conn.close()
        if row and row['contributors'] and row['contributors'] > 0:
            contributors = row['contributors']
            # Validado si >= 2 DJs con ajustes similares
            validated = contributors >= 2
            return {
                'bpm_adjust': round(row['bpm_adj'] or 0.0, 4),
                'beat_offset': round(row['beat_off'] or 0.0, 6),
                'contributors': contributors,
                'validated': validated,
                'original_bpm': round(row['orig_bpm'] or 0.0, 2),
            }
        return {
            'bpm_adjust': 0.0,
            'beat_offset': 0.0,
            'contributors': 0,
            'validated': False,
        }

    # ==================== AUDD AUTO-TRIGGER LOG ====================

    def log_audd_call(self, fingerprint: str, success: bool,
                      artist: Optional[str] = None,
                      title: Optional[str] = None) -> None:
        """Registra una llamada AudD (incluso fallidas) para honrar cooldown y cap."""
        if not fingerprint:
            return
        conn = self._open_conn()
        try:
            conn.execute(
                'INSERT INTO audd_call_log (fingerprint, called_at, success, artist, title) '
                'VALUES (?, ?, ?, ?, ?)',
                (fingerprint, time.time(), 1 if success else 0, artist, title),
            )
            conn.commit()
        finally:
            conn.close()

    def get_last_audd_call(self, fingerprint: str) -> Optional[float]:
        """Devuelve timestamp UNIX de la ultima llamada AudD para este fingerprint, o None."""
        if not fingerprint:
            return None
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT MAX(called_at) AS last FROM audd_call_log WHERE fingerprint = ?',
                (fingerprint,),
            )
            row = c.fetchone()
            return row['last'] if row and row['last'] else None
        finally:
            conn.close()

    def count_audd_calls_today(self) -> int:
        """Cuenta llamadas AudD del dia UTC actual."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT COUNT(*) AS n FROM audd_call_log WHERE called_at >= ?',
                (today_start,),
            )
            row = c.fetchone()
            return row['n'] if row else 0
        finally:
            conn.close()
