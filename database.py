import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Optional

class AnalysisDB:
    def __init__(self, db_path="analysis.db"):
        self.db_path = db_path
        self._conn = None  # Conexión persistente
        self.init_db()
    
    @property
    def conn(self):
        """Propiedad que retorna una conexión a la BD (lazy loading)"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Tabla de análisis
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
                fingerprint TEXT
            )
        ''')
        
        # Crear índices para búsquedas rápidas
        c.execute('CREATE INDEX IF NOT EXISTS idx_artist ON tracks(artist)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_genre ON tracks(genre)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_bpm ON tracks(bpm)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_energy ON tracks(energy_dj)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_key ON tracks(key)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_camelot ON tracks(camelot)')
        
        # Tabla de correcciones manuales (memoria colectiva)
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
        
        # Tabla de notas DJ
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
        
        conn.commit()
        conn.close()

    def _row_to_dict(self, row) -> Optional[Dict]:
        if not row:
            return None
        
        base_dict = {
            'id': row[0],
            'filename': row[1],
            'artist': row[2],
            'title': row[3],
            'duration': row[4],
            'bpm': row[5],
            'key': row[6],
            'camelot': row[7],
            'energy_dj': row[8],
            'genre': row[9],
            'track_type': row[10],
            'analysis_json': row[11],
            'analyzed_at': row[12],
            'fingerprint': row[13]
        }
        
        # Extraer campos adicionales del analysis_json (artwork_url, label, etc.)
        if row[11]:  # analysis_json
            try:
                full_analysis = json.loads(row[11])
                # Añadir campos importantes que no están en las columnas
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
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM tracks WHERE filename = ?', (filename,))
        result = c.fetchone()
        conn.close()
        return result
    
    def get_track_by_id(self, track_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM tracks WHERE id = ?', (track_id,))
        result = c.fetchone()
        conn.close()
        return self._row_to_dict(result)
    
    def save_track(self, track_data):
        conn = sqlite3.connect(self.db_path)
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
    
    def save_correction(self, track_id, field, old_value, new_value, fingerprint=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO corrections 
            (track_id, field, old_value, new_value, corrected_at, fingerprint)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (track_id, field, old_value, new_value, datetime.now().isoformat(), fingerprint))
        
        conn.commit()
        conn.close()
    
    def get_collective_genre(self, fingerprint):
        if not fingerprint:
            return None
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT new_value, COUNT(*) as count
            FROM corrections
            WHERE fingerprint = ? AND field = 'genre'
            GROUP BY new_value
            ORDER BY count DESC
            LIMIT 1
        ''', (fingerprint,))
        
        result = c.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    # ==================== BÚSQUEDAS ====================
    
    def search_by_artist_title(self, artist: str, title: str):
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
    
    def search_advanced(self, artist=None, genre=None, min_bpm=None, max_bpm=None,
                       min_energy=None, max_energy=None, key=None, track_type=None, limit=100):
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM tracks ORDER BY analyzed_at DESC LIMIT ?', (limit,))
        results = c.fetchall()
        conn.close()
        return self._rows_to_list(results)
    
    def get_unique_artists(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT DISTINCT artist FROM tracks 
            WHERE artist IS NOT NULL AND artist != ''
            ORDER BY artist COLLATE NOCASE
        ''')
        results = [row[0] for row in c.fetchall()]
        conn.close()
        return results
    
    def get_unique_genres(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT DISTINCT genre FROM tracks 
            WHERE genre IS NOT NULL AND genre != ''
            ORDER BY genre COLLATE NOCASE
        ''')
        results = [row[0] for row in c.fetchall()]
        conn.close()
        return results
    
    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM tracks')
        total_tracks = c.fetchone()[0]
        
        c.execute('SELECT COUNT(DISTINCT artist) FROM tracks WHERE artist IS NOT NULL')
        unique_artists = c.fetchone()[0]
        
        c.execute('SELECT COUNT(DISTINCT genre) FROM tracks WHERE genre IS NOT NULL')
        unique_genres = c.fetchone()[0]
        
        c.execute('SELECT AVG(bpm) FROM tracks')
        avg_bpm = c.fetchone()[0] or 0
        
        c.execute('SELECT AVG(energy_dj) FROM tracks')
        avg_energy = c.fetchone()[0] or 0
        
        c.execute('SELECT SUM(duration) FROM tracks')
        total_duration = c.fetchone()[0] or 0
        
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
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO dj_notes (track_id, fingerprint, note, created_at)
            VALUES (?, ?, ?, ?)
        ''', (track_id, fingerprint, note, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def get_dj_notes(self, track_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT note, created_at FROM dj_notes
            WHERE track_id = ? ORDER BY created_at DESC
        ''', (track_id,))
        results = [{'note': row[0], 'created_at': row[1]} for row in c.fetchall()]
        conn.close()
        return results
    
    def delete_track(self, track_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM tracks WHERE id = ?', (track_id,))
        deleted = c.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
