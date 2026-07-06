import sqlite3
import os
import threading
import time
from datetime import datetime, timezone
import json
import re
import statistics
from typing import List, Dict, Optional


def _scrub_volatile_tokens(text: str) -> str:
    """Sustituye tokens VARIABLES de un mensaje de error por placeholders
    estables, para que dos errores del MISMO bug logico que solo difieren en
    el path/hash/numero concreto produzcan la misma clave de agrupacion.

    Sin esto, errores de servidor como
      "NoBackendError: ... for /tmp/u0.mp3"
      "NoBackendError: ... for /tmp/u1.mp3"
    generaban un grupo distinto por cada archivo -> decenas de singletons que
    en la UI se ven identicos (la UI muestra un human_message truncado) y que
    al resolver uno dejaban vivos a los gemelos => "no desaparece".

    Orden importa: paths antes que numeros (un path puede contener digitos).
    """
    # Rutas Windows (C:\foo\bar) y POSIX (/tmp/foo/bar.mp3). Captura tambien
    # la extension. Se hace ANTES de scrubbear numeros/hex.
    text = re.sub(r'[A-Za-z]:\\[^\s\'"]+', '<path>', text)
    text = re.sub(r'(?:/[\w.\-]+){2,}', '<path>', text)
    # Direcciones de memoria y hex largos (md5/sha/fingerprints).
    text = re.sub(r'0x[0-9a-fA-F]+', '<addr>', text)
    text = re.sub(r'\b[0-9a-fA-F]{8,}\b', '<hex>', text)
    # UUIDs.
    text = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<uuid>', text)
    # Numeros de 2+ digitos (line numbers, tamanos, puertos, timestamps).
    # 2+ para no tocar cosas como "0-dimensional" (un solo digito).
    text = re.sub(r'\b\d{2,}\b', '<n>', text)
    return text


def normalize_error_key(error_msg: Optional[str]) -> str:
    """Clave de agrupacion estable de un error: el mensaje SIN el prefijo
    '[plataforma version]' que anade /client-error, primera linea, tokens
    volatiles scrubbeados (ver _scrub_volatile_tokens), 80 chars.

    Critico para que "[ios 2.9.0] X" y "[ios 2.9.2] X" caigan en el MISMO
    grupo. Antes se agrupaba por substr(error_msg,1,80) crudo, que metia la
    version en la clave -> el mismo bug logico aparecia como N grupos (uno
    por version) que en la UI se veian identicos (la UI muestra clean_msg).
    Resolver uno dejaba los gemelos de otras versiones -> "no desaparece".

    Ademas scrubbea paths/hash/numeros: errores de servidor que solo difieren
    en el archivo concreto (/tmp/aX.mp3) colapsan en un unico grupo en vez de
    generar decenas de singletons identicos a la vista.
    """
    raw = error_msg or ''
    m = re.match(r'^\[([^\]]*)\]\s*(.*)$', raw, re.DOTALL)
    clean = m.group(2) if m else raw
    clean = clean.strip()
    if not clean:
        return ''
    first_line = clean.splitlines()[0]
    return _scrub_volatile_tokens(first_line)[:80]


def derive_error_meta(error_class: str, error_msg: Optional[str],
                      endpoint: Optional[str]) -> Dict[str, Optional[str]]:
    """Deriva metadatos FIABLES de un error a partir de sus campos crudos.

    El `endpoint` es el discriminador fiable cliente/servidor (lo setea el
    backend al persistir): 'client:*' = error reportado por la app Flutter;
    'unhandled:*' / '/analyze' / 'preview' / 'artwork' / '/identify' =
    servidor. El TEXTO del mensaje NO se usa para decidir el origen — esa fue
    justo la causa de los mal-etiquetados historicos (un FileSystemException de
    iOS acababa rotulado como "disco en Render").

    Devuelve origin, platform, app_version, context y clean_msg (el mensaje sin
    el prefijo "[plataforma version]" que anade /client-error), mas un
    human_message SIEMPRE atribuido al origen correcto.
    """
    endpoint = endpoint or ''
    raw = error_msg or ''
    origin = 'client' if endpoint.startswith('client:') else 'server'

    # Errores de cliente vienen con prefijo "[plataforma version] msg".
    platform: Optional[str] = None
    app_version: Optional[str] = None
    clean_msg = raw
    m = re.match(r'^\[([^\]]*)\]\s*(.*)$', raw, re.DOTALL)
    if m:
        tag = m.group(1).split()
        clean_msg = m.group(2)
        if tag:
            platform = tag[0]
            if len(tag) > 1:
                app_version = tag[1]

    context = endpoint.split(':', 1)[1] if ':' in endpoint else endpoint

    where = (
        f"cliente {platform}" if origin == 'client' and platform
        else 'cliente' if origin == 'client'
        else 'servidor'
    )
    summary = (clean_msg or error_class).strip().splitlines()[0][:80]
    human_message = f"[{where}] {summary}" if summary else f"[{where}] {error_class}"

    return {
        'origin': origin,
        'platform': platform,
        'app_version': app_version,
        'context': context,
        'clean_msg': clean_msg,
        'human_message': human_message,
    }

class AnalysisDB:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "/data/analysis.db")
        self.db_path = db_path
        # Conexion persistente POR HILO. Ver la property `conn`.
        self._local = threading.local()
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
        # timeout=30 + busy_timeout: si otra conexion tiene la BD bloqueada,
        # esperamos hasta 30s en vez de fallar al instante con "database is
        # locked". WAL permite lectores concurrentes con un escritor; el
        # busy_timeout cubre la contienda entre escritores (save_track,
        # delete_track_by_filename, corrections) bajo carga concurrente.
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def conn(self):
        """Conexion persistente POR HILO con WAL + row_factory (lazy).

        ANTES era UNA conexion compartida (`self._conn`) con
        check_same_thread=False. Funcionaba solo porque todos sus usuarios
        (community cues, search, search_collective_db) viven en el event loop
        async, que los serializa. Pero el flag silencia el guard de sqlite3:
        el dia que alguien moviera un endpoint que usa `db.conn` a un
        threadpool (como ya hace /analyze con `analyze_audio`), o llamara a
        `db.conn` desde el hilo worker, el MISMO objeto Connection se usaria
        desde dos hilos -> corrupcion/errores silenciosos ("recursive use of
        cursors", transacciones entrelazadas).

        Thread-local da a cada hilo su propia conexion al mismo fichero WAL
        (el patron concurrente recomendado por SQLite: N conexiones + WAL +
        busy_timeout). Asi `db.conn` es seguro venga del hilo que venga, sin
        cambiar el comportamiento actual (el event loop sigue reutilizando una
        unica conexion de larga vida). El numero de hilos del threadpool es
        acotado, asi que las conexiones no crecen sin limite.
        """
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

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
                chromaprint TEXT,
                acoustic_id TEXT,
                analysis_version TEXT
            )
        ''')

        # Migracion: anadir columna chromaprint si no existe (BDs antiguas)
        try:
            c.execute('ALTER TABLE tracks ADD COLUMN chromaprint TEXT')
        except sqlite3.OperationalError:
            pass  # Ya existe

        # Migracion: acoustic_id = id del CLUSTER ACUSTICO al que pertenece el
        # track (memoria colectiva por sonido, no por bytes). NULL en tracks
        # analizados antes de esta feature; se rellena al re-analizar. Ver
        # resolve_acoustic_cluster / canonical_community_key.
        try:
            c.execute('ALTER TABLE tracks ADD COLUMN acoustic_id TEXT')
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
        # idx_acoustic: lookup O(1) del cluster por fingerprint canonico +
        # candidatos para el barrido Hamming acotado por duracion.
        c.execute('CREATE INDEX IF NOT EXISTS idx_acoustic ON tracks(acoustic_id)')

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

        # Migracion: BDs creadas antes de v2.9.3 tienen beat_grid_corrections
        # SIN created_at/updated_at (el CREATE IF NOT EXISTS no altera tablas
        # ya existentes). El INSERT de save_beat_grid_correction referencia
        # ambas columnas -> 500 "no column named created_at". ALTER idempotente.
        for _bg_col in ('created_at', 'updated_at'):
            try:
                conn.execute(f'ALTER TABLE beat_grid_corrections ADD COLUMN {_bg_col} TEXT')
            except sqlite3.OperationalError:
                pass  # Ya existe

        # Migracion (2026-07-04): asegurar UNIQUE(fingerprint, device_id) via
        # INDICE. Mismo patron que las columnas de arriba: las BDs creadas antes
        # de que el CREATE llevara el `UNIQUE(fingerprint, device_id)` inline
        # tienen la tabla SIN esa constraint (CREATE IF NOT EXISTS no altera
        # tablas existentes). El INSERT ... ON CONFLICT(fingerprint, device_id)
        # de submit_beat_grid_correction NO encuentra constraint que casar ->
        # sqlite3.OperationalError -> 500 en CADA POST /community/beat-grid
        # (visto en produccion 2026-07-04, mismo device reintentando). Un UNIQUE
        # INDEX satisface el ON CONFLICT igual que la constraint inline y es
        # idempotente (no-op si la tabla ya la tiene). Deduplicamos primero por
        # si la tabla vieja, sin la constraint, acumulo (fingerprint, device_id)
        # repetidos que harian fallar el CREATE UNIQUE INDEX.
        try:
            conn.execute('''
                DELETE FROM beat_grid_corrections
                WHERE id NOT IN (
                    SELECT MAX(id) FROM beat_grid_corrections
                    GROUP BY fingerprint, device_id
                )
            ''')
            conn.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_beat_grid_fp_device
                ON beat_grid_corrections(fingerprint, device_id)
            ''')
        except sqlite3.OperationalError:
            pass

        # Track type community overrides (Fase 2 v2).
        # Un device_id puede votar 1 vez por fingerprint. Si N>=3 votos y el
        # winner supera al 2do por >=2, ese tipo gana sobre el algoritmico.
        # Implementacion en `get_track_type_consensus`.
        #
        # DEPRECADO Fase 4: reemplazado por community_overrides (tabla
        # generica con `field` column). Se mantiene viva temporalmente para
        # migrar datos historicos y para que upgrades en caliente no pierdan
        # votos. Helpers nuevos escriben SOLO a community_overrides; los
        # legacy wrappers (submit_track_type_override, etc.) tambien.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS community_track_type_overrides (
                fingerprint TEXT NOT NULL,
                device_id TEXT NOT NULL,
                track_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (fingerprint, device_id)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_cttov_fp ON community_track_type_overrides(fingerprint)')

        # Community overrides genericos (Fase 4 v2).
        # Sistema unificado de votos comunitarios para CUALQUIER campo
        # categorico (track_type, key, camelot, genre, subgenre). Mismas
        # reglas de consensus: >=3 votos al winner y supera al 2do por >=2.
        # `field` y `value` siempre strings; el caller los normaliza/valida
        # segun whitelist por campo (COMMUNITY_VALID_VALUES en main.py).
        conn.execute('''
            CREATE TABLE IF NOT EXISTS community_overrides (
                fingerprint TEXT NOT NULL,
                device_id TEXT NOT NULL,
                field TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (fingerprint, device_id, field)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_co_fp_field ON community_overrides(fingerprint, field)')
        # Migracion idempotente: si la tabla legacy tiene votos y la nueva
        # esta vacia para ese fp, copiar como field='track_type'. Sin LOCK
        # porque ON CONFLICT DO NOTHING garantiza idempotencia si se llama
        # multiples veces (ej. reinicio de Render).
        conn.execute('''
            INSERT OR IGNORE INTO community_overrides
                (fingerprint, device_id, field, value, created_at)
            SELECT fingerprint, device_id, 'track_type', track_type, created_at
            FROM community_track_type_overrides
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

        # Tabla de errores de analisis (privacy-first: filename hasheado).
        # Captura fallos de /analyze y /identify para diagnostico operacional
        # desde el panel admin. NO se guarda el filename real, solo su MD5;
        # device_id sirve para drill-down por usuario sin exponer contenido.
        # Schema 1.
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                device_id TEXT,
                filename_hash TEXT,
                fingerprint TEXT,
                error_class TEXT NOT NULL,
                error_msg TEXT,
                traceback TEXT,
                endpoint TEXT DEFAULT '/analyze',
                resolved INTEGER NOT NULL DEFAULT 0,
                resolved_at TEXT,
                msg_short TEXT GENERATED ALWAYS AS (substr(COALESCE(error_msg,''),1,80)) VIRTUAL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_aerr_ts ON analysis_errors(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_aerr_resolved ON analysis_errors(resolved)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_aerr_device ON analysis_errors(device_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_aerr_class ON analysis_errors(error_class)')

        # Engine source: que motor analizo este track (render | local_engine).
        # Permite calcular ratios desde el panel admin. ALTER TABLE idempotente
        # para no romper BDs antiguas. Default NULL = origen desconocido
        # (tracks pre-instrumentacion).
        try:
            conn.execute("ALTER TABLE tracks ADD COLUMN engine_source TEXT")
        except sqlite3.OperationalError:
            # Columna ya existe.
            pass
        conn.execute('CREATE INDEX IF NOT EXISTS idx_tracks_engine ON tracks(engine_source)')
        # Index para el feed de actividad en vivo del panel admin
        # (ORDER BY analyzed_at DESC LIMIT N + COUNT por ventana de tiempo).
        # El endpoint /admin/activity es de auto-refresh, asi que la query es
        # "caliente" y conviene tenerla indexada.
        conn.execute('CREATE INDEX IF NOT EXISTS idx_tracks_analyzed_at ON tracks(analyzed_at)')

        # Versión del motor de análisis. NULL = "1" (tracks pre-versionado).
        # Incrementar ANALYSIS_VERSION en main.py invalida la cache y fuerza
        # re-análisis automático en la próxima subida/consulta.
        try:
            conn.execute("ALTER TABLE tracks ADD COLUMN analysis_version TEXT")
        except sqlite3.OperationalError:
            pass

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

    def resolve_acoustic_cluster(self, raw_ints, duration=None):
        """Devuelve el `acoustic_id` (cluster acustico) de un fingerprint.

        Busca entre los tracks ya vistos uno cuyo Chromaprint este dentro del
        umbral Hamming (mismo audio, aunque sea otro codec/bitrate/tag). Si lo
        encuentra, el track nuevo hereda ESE cluster -> comparten memoria
        colectiva. Si no, arranca un cluster nuevo cuyo id es la clave exacta
        (MD5 del array) del primer miembro.

        El barrido se acota a tracks de duracion similar (+-2.5s): el mismo
        audio dura lo mismo, asi que esto reduce O(N) a O(pocos) sin perder
        matches. Best-effort: `raw_ints` None/vacio -> None (sin cluster).
        """
        from acoustic_fingerprint import (
            MATCH_THRESHOLD, acoustic_key, decode_raw, hamming_distance,
        )
        if not raw_ints:
            return None

        conn = self._open_conn()
        c = conn.cursor()
        if duration is not None and duration > 0:
            c.execute(
                'SELECT chromaprint, acoustic_id FROM tracks '
                'WHERE chromaprint IS NOT NULL AND acoustic_id IS NOT NULL '
                'AND duration IS NOT NULL AND ABS(duration - ?) <= 2.5',
                (duration,),
            )
        else:
            c.execute(
                'SELECT chromaprint, acoustic_id FROM tracks '
                'WHERE chromaprint IS NOT NULL AND acoustic_id IS NOT NULL'
            )
        rows = c.fetchall()
        conn.close()

        best_id, best_dist = None, MATCH_THRESHOLD
        for row in rows:
            cand = decode_raw(row['chromaprint'])
            if not cand:
                continue
            d = hamming_distance(raw_ints, cand)
            if d < best_dist:
                best_dist, best_id = d, row['acoustic_id']

        return best_id if best_id is not None else acoustic_key(raw_ints)

    def canonical_community_key(self, fingerprint):
        """Normaliza el fingerprint que llega a un endpoint /community/* a la
        clave del CLUSTER ACUSTICO, para que todas las versiones del mismo
        audio (mp3/flac/otro tag) compartan memoria colectiva.

        Cae al fingerprint original si el track no esta en BD o no tiene
        acoustic_id (compat total con datos previos a esta feature: la memoria
        vieja, guardada bajo el fingerprint MD5, sigue siendo accesible).
        """
        if not fingerprint:
            return fingerprint
        conn = self._open_conn()
        c = conn.cursor()
        c.execute(
            'SELECT acoustic_id FROM tracks WHERE fingerprint = ? OR id = ? '
            'LIMIT 1',
            (fingerprint, fingerprint),
        )
        row = c.fetchone()
        conn.close()
        if row and row['acoustic_id']:
            return row['acoustic_id']
        return fingerprint

    def canonical_community_keys(self, fingerprints):
        """Version batch de canonical_community_key: mapa {fingerprint ->
        clave del cluster} para una lista, en UNA query. Los que no matchean un
        track con acoustic_id se mapean a si mismos (fallback). Lo usan los
        endpoints batch (columnas de popularidad/rating de la libreria desktop)
        para no disparar N queries."""
        fps = [f for f in (fingerprints or []) if f]
        if not fps:
            return {}
        out = {f: f for f in fps}  # fallback: cada uno a si mismo
        want = set(fps)
        placeholders = ','.join('?' * len(fps))
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT fingerprint, id, acoustic_id FROM tracks '
                'WHERE acoustic_id IS NOT NULL AND '
                f'(fingerprint IN ({placeholders}) OR id IN ({placeholders}))',
                fps + fps,
            )
            for row in c.fetchall():
                aid = row['acoustic_id']
                if row['fingerprint'] in want:
                    out[row['fingerprint']] = aid
                if row['id'] in want:
                    out[row['id']] = aid
        finally:
            conn.close()
        return out

    def fingerprints_in_cluster(self, fingerprint):
        """Todos los fingerprints (y track ids) del mismo cluster acustico que
        `fingerprint`, incluido el propio. Sirve para AGREGAR la memoria
        colectiva de TODAS las versiones del mismo audio en la LECTURA, sin
        tocar la escritura (usado por los cues comunitarios, que comparten tabla
        con el sync personal y por eso no se pueden re-key en escritura).

        Si el track no tiene cluster todavia, devuelve solo el propio
        fingerprint (comportamiento identico al de antes de la feature).
        """
        if not fingerprint:
            return []
        keys = {fingerprint}
        conn = self._open_conn()
        try:
            c = conn.cursor()
            # Cluster del fingerprint pedido (por fingerprint o por id).
            c.execute(
                'SELECT acoustic_id FROM tracks WHERE fingerprint = ? OR id = ? '
                'LIMIT 1',
                (fingerprint, fingerprint),
            )
            row = c.fetchone()
            aid = row['acoustic_id'] if row else None
            if aid:
                c.execute(
                    'SELECT fingerprint, id FROM tracks WHERE acoustic_id = ?',
                    (aid,),
                )
                for r in c.fetchall():
                    if r['fingerprint']:
                        keys.add(r['fingerprint'])
                    if r['id']:
                        keys.add(r['id'])
        finally:
            conn.close()
        return list(keys)

    def save_track(self, track_data):
        conn = self._open_conn()
        c = conn.cursor()

        # `chromaprint` es un blob tecnico (base64 de varios KB) cuyo sitio es
        # su columna dedicada; fuera del analysis_json para no duplicarlo ni
        # engordar lo que se envia al cliente ni el AnalysisResult del cache-hit.
        analysis_json_data = {
            k: v for k, v in track_data.items() if k != 'chromaprint'
        }

        c.execute('''
            INSERT OR REPLACE INTO tracks
            (id, filename, artist, title, duration, bpm, key, camelot,
             energy_dj, genre, track_type, analysis_json, analyzed_at,
             fingerprint, chromaprint, acoustic_id, engine_source,
             analysis_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(analysis_json_data),
            datetime.now().isoformat(),
            track_data.get('fingerprint'),
            track_data.get('chromaprint'),
            track_data.get('acoustic_id'),
            track_data.get('engine_source'),
            track_data.get('analysis_version'),
        ))

        conn.commit()
        conn.close()

    def save_correction(self, track_id, field, old_value, new_value, fingerprint=None, device_id=None):
        # Memoria colectiva por SONIDO: agrupar bajo el cluster acustico.
        fingerprint = self.canonical_community_key(fingerprint)
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
        fingerprint = self.canonical_community_key(fingerprint)

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
        fingerprint = self.canonical_community_key(fingerprint)

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
        fingerprint = self.canonical_community_key(fingerprint)
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
        fingerprint = self.canonical_community_key(fingerprint)
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
        fingerprint = self.canonical_community_key(fingerprint)
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
        fingerprint = self.canonical_community_key(fingerprint)
        conn = self._open_conn()
        c = conn.cursor()
        if rating <= 0:
            # rating 0 = el DJ QUITA su valoracion (toggle off en la UI).
            c.execute('DELETE FROM track_ratings WHERE fingerprint = ? AND device_id = ?',
                      (fingerprint, device_id))
        else:
            c.execute('''
                INSERT INTO track_ratings (fingerprint, device_id, rating, rated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(fingerprint, device_id) DO UPDATE SET rating = ?, rated_at = ?
            ''', (fingerprint, device_id, rating, datetime.utcnow().isoformat(),
                  rating, datetime.utcnow().isoformat()))
        # Recalcular media (con count=0 -> avg NULL -> 0)
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
        fingerprint = self.canonical_community_key(fingerprint)
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

    def get_track_popularity_batch(self, fingerprints: List[str]) -> Dict[str, Dict]:
        """Popularidad de varios fingerprints en UNA query (columna de libreria
        del cliente desktop). Devuelve {fingerprint: {analysis_count, dj_count,
        avg_rating, total_ratings}}. Los que no esten en la tabla NO aparecen
        (el cliente asume 0). Acota a 500 por si el cliente no trocea."""
        fps = [f for f in (fingerprints or []) if f][:500]
        if not fps:
            return {}
        # Memoria colectiva por sonido: consultar por el cluster acustico, pero
        # devolver keyed por el fingerprint ORIGINAL que mando el cliente.
        canon = self.canonical_community_keys(fps)          # {orig -> cluster}
        keys = list({canon[f] for f in fps})
        placeholders = ','.join('?' * len(keys))
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT fingerprint, analysis_count, dj_count, avg_rating, '
                'total_ratings FROM track_popularity '
                f'WHERE fingerprint IN ({placeholders})',
                keys,
            )
            by_key: Dict[str, Dict] = {}
            for row in c.fetchall():
                by_key[row['fingerprint']] = {
                    'analysis_count': row['analysis_count'] or 0,
                    'dj_count': row['dj_count'] or 0,
                    'avg_rating': round(row['avg_rating'] or 0, 1),
                    'total_ratings': row['total_ratings'] or 0,
                }
            return {f: by_key[canon[f]] for f in fps if canon[f] in by_key}
        finally:
            conn.close()

    def get_my_rating(self, fingerprint: str, device_id: str) -> int:
        fingerprint = self.canonical_community_key(fingerprint)
        conn = self._open_conn()
        c = conn.cursor()
        c.execute('SELECT rating FROM track_ratings WHERE fingerprint = ? AND device_id = ?',
                  (fingerprint, device_id))
        row = c.fetchone()
        conn.close()
        return row['rating'] if row else 0

    def get_my_ratings_batch(self, fingerprints: List[str], device_id: str) -> Dict[str, int]:
        """Rating PROPIO (de este device_id) de varios tracks en UNA query.
        Lo usa la columna de rating personal de la librería desktop. Devuelve
        {fingerprint: rating}. Los tracks sin valorar por este device NO
        aparecen (el cliente asume 0). Acota a 500 por seguridad."""
        fps = [f for f in (fingerprints or []) if f][:500]
        if not fps or not device_id:
            return {}
        # Query por el cluster acustico, devolver keyed por el original.
        canon = self.canonical_community_keys(fps)          # {orig -> cluster}
        keys = list({canon[f] for f in fps})
        placeholders = ','.join('?' * len(keys))
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT fingerprint, rating FROM track_ratings '
                f'WHERE device_id = ? AND fingerprint IN ({placeholders})',
                [device_id] + keys,
            )
            by_key = {row['fingerprint']: row['rating'] for row in c.fetchall()}
            return {f: by_key[canon[f]] for f in fps if canon[f] in by_key}
        finally:
            conn.close()

    # ==================== COMMUNITY BEAT GRID ====================

    def submit_beat_grid_correction(self, fingerprint: str, device_id: str,
                                     bpm_adjust: float, beat_offset: float,
                                     original_bpm: float):
        """Guarda o actualiza la correccion de beat grid de un DJ"""
        fingerprint = self.canonical_community_key(fingerprint)
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
        fingerprint = self.canonical_community_key(fingerprint)
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

    # ==================== COMMUNITY OVERRIDES GENERICOS (Fase 4) ====================
    # Sistema unificado para CUALQUIER campo categorico: track_type, key,
    # camelot, genre, subgenre. Mismas reglas de consensus (>=3 votos al
    # winner, supera al 2do por >=2). Whitelist por campo en main.py.

    def submit_community_override(
        self, fingerprint: str, device_id: str, field: str, value: str,
    ) -> None:
        """Voto/cambia voto de un device sobre un campo de un track.

        PK (fingerprint, device_id, field) -> un device puede votar 1 campo
        1 vez por track; segundo POST sobreescribe (cambio de opinion).
        """
        fingerprint = self.canonical_community_key(fingerprint)
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        conn = self._open_conn()
        try:
            conn.execute('''
                INSERT INTO community_overrides
                    (fingerprint, device_id, field, value, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint, device_id, field) DO UPDATE SET
                    value = excluded.value,
                    created_at = excluded.created_at
            ''', (fingerprint, device_id, field, value, now))
            conn.commit()
        finally:
            conn.close()

    def get_community_consensus(
        self, fingerprint: str, field: str,
    ) -> Optional[Dict]:
        """Devuelve consensus si los votos son inequivocos para (fp, field).

        Reglas (mismas que Fase 2):
          - >= 3 votos totales al winner.
          - winner supera al 2do por >= 2 votos.
        """
        fingerprint = self.canonical_community_key(fingerprint)
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute('''
                SELECT value, COUNT(*) AS votes
                FROM community_overrides
                WHERE fingerprint = ? AND field = ?
                GROUP BY value
                ORDER BY votes DESC
            ''', (fingerprint, field))
            rows = c.fetchall()
        finally:
            conn.close()

        if not rows:
            return None

        distribution = {r['value']: r['votes'] for r in rows}
        total = sum(distribution.values())
        winner_value = rows[0]['value']
        winner_votes = rows[0]['votes']
        second_votes = rows[1]['votes'] if len(rows) > 1 else 0

        if winner_votes < 3:
            return None
        if winner_votes - second_votes < 2:
            return None

        return {
            'value': winner_value,
            'votes': winner_votes,
            'total': total,
            'distribution': distribution,
        }

    def get_community_consensus_numeric(
        self, fingerprint: str, field: str, threshold: int = 3,
    ) -> Dict:
        """Calcula consensus numerico (Fase 5) usando MEDIANA.

        A diferencia del consensus categorico (Fase 4) que usa MODA + tiebreak,
        este metodo asume que los valores son numericos y calcula la mediana
        sobre todos los votos del campo. La normalizacion previa la hace el
        endpoint POST (ej. BPM colapsado al rango [60, 180] via bpm_utils).

        Args:
            fingerprint: hash del track
            field: 'bpm' o 'energy'
            threshold: minimo de votos para tener consensus (default 3)

        Returns:
            {
                'consensus': float | int | None,  # mediana si N >= threshold
                'consensus_votes': int,            # = total_voters si hay consensus
                'votes_distribution': {value_str: count},
                'total_voters': int
            }

        Notas:
        - Para `bpm` redondeamos a 1 decimal (alineado con bpm_utils).
        - Para `energy` redondeamos a entero (escala DJ 1-10).
        - Si algun valor no parsea a float se ignora (defensivo, los votos
          deberian estar normalizados por _validate_community_field).
        """
        fingerprint = self.canonical_community_key(fingerprint)
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute('''
                SELECT value, COUNT(*) AS votes
                FROM community_overrides
                WHERE fingerprint = ? AND field = ?
                GROUP BY value
                ORDER BY votes DESC
            ''', (fingerprint, field))
            rows = c.fetchall()
        finally:
            conn.close()

        distribution = {r['value']: r['votes'] for r in rows}
        total_voters = sum(distribution.values())

        if total_voters < threshold:
            return {
                'consensus': None,
                'consensus_votes': 0,
                'votes_distribution': distribution,
                'total_voters': total_voters,
            }

        # Expandir a lista de floats respetando los conteos.
        flat_values: List[float] = []
        for value_str, count in distribution.items():
            try:
                parsed = float(value_str)
            except (TypeError, ValueError):
                # Voto malformado en BD: ignorar pero no romper.
                continue
            flat_values.extend([parsed] * int(count))

        if not flat_values or len(flat_values) < threshold:
            return {
                'consensus': None,
                'consensus_votes': 0,
                'votes_distribution': distribution,
                'total_voters': total_voters,
            }

        median_value = statistics.median(flat_values)
        if field == 'bpm':
            consensus_value: Any = round(float(median_value), 1)
        elif field == 'energy':
            consensus_value = int(round(median_value))
        else:
            consensus_value = round(float(median_value), 2)

        return {
            'consensus': consensus_value,
            'consensus_votes': total_voters,
            'votes_distribution': distribution,
            'total_voters': total_voters,
        }

    def get_community_votes(self, fingerprint: str, field: str) -> Dict:
        """Distribucion bruta de votos por (fp, field). Siempre devuelve dict."""
        fingerprint = self.canonical_community_key(fingerprint)
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute('''
                SELECT value, COUNT(*) AS votes
                FROM community_overrides
                WHERE fingerprint = ? AND field = ?
                GROUP BY value
                ORDER BY votes DESC
            ''', (fingerprint, field))
            rows = c.fetchall()
        finally:
            conn.close()
        return {r['value']: r['votes'] for r in rows}

    def delete_community_override(
        self, fingerprint: str, device_id: str, field: str,
    ) -> bool:
        """Retira el voto de un device sobre (fingerprint, field).

        Idempotente: si el voto no existia, devuelve False (no fue eliminado
        pero no es un error). Asi el cliente puede llamar 'retirar voto' sin
        chequear previamente si voto.

        Returns:
            True si se elimino un row, False si no habia voto previo.
        """
        fingerprint = self.canonical_community_key(fingerprint)
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute('''
                DELETE FROM community_overrides
                WHERE fingerprint = ? AND device_id = ? AND field = ?
            ''', (fingerprint, device_id, field))
            deleted = c.rowcount > 0
            conn.commit()
            return deleted
        finally:
            conn.close()

    # ==================== TRACK TYPE WRAPPERS (Fase 2 backwards-compat) ====================
    # Mantenidos como wrappers de los genéricos arriba. Asi el codigo legacy
    # de main.py + clientes Flutter pre-Fase 4 siguen funcionando sin cambios.

    def submit_track_type_override(self, fingerprint: str, device_id: str, track_type: str) -> None:
        """Wrapper legacy: delega a submit_community_override(field='track_type')."""
        self.submit_community_override(fingerprint, device_id, 'track_type', track_type)

    def get_track_type_consensus(self, fingerprint: str) -> Optional[Dict]:
        """Wrapper legacy: delega a get_community_consensus(field='track_type').

        Reformatea la respuesta para mantener la shape historica de Fase 2
        (key 'type' en lugar de 'value').
        """
        consensus = self.get_community_consensus(fingerprint, 'track_type')
        if not consensus:
            return None
        return {
            'type': consensus['value'],
            'votes': consensus['votes'],
            'total': consensus['total'],
            'distribution': consensus['distribution'],
        }

    def get_track_type_votes(self, fingerprint: str) -> Dict:
        """Wrapper legacy: delega a get_community_votes(field='track_type')."""
        return self.get_community_votes(fingerprint, 'track_type')

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

    # ─────────────── analysis_errors helpers ───────────────

    def log_analysis_error(
        self,
        *,
        device_id: Optional[str],
        filename: Optional[str],
        fingerprint: Optional[str],
        error_class: str,
        error_msg: str,
        traceback_str: Optional[str] = None,
        endpoint: str = '/analyze',
    ) -> int:
        """Registra un error de analisis con filename HASHEADO (privacy).

        device_id, fingerprint y traceback opcionales (algunos fallos
        ocurren antes de tener el fp). error_msg se trunca a 1000 chars
        para evitar payloads gigantes.
        """
        import hashlib
        filename_hash = (
            hashlib.md5(filename.encode('utf-8', errors='replace')).hexdigest()
            if filename else None
        )
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'INSERT INTO analysis_errors '
                '(device_id, filename_hash, fingerprint, error_class, '
                ' error_msg, traceback, endpoint) '
                'VALUES (?,?,?,?,?,?,?)',
                (
                    device_id,
                    filename_hash,
                    fingerprint,
                    error_class[:120],
                    (error_msg or '')[:1000],
                    (traceback_str or '')[:4000] if traceback_str else None,
                    endpoint,
                ),
            )
            conn.commit()
            return c.lastrowid or 0
        except sqlite3.OperationalError:
            # Tabla no creada todavia en BDs muy antiguas.
            return 0
        finally:
            conn.close()

    def get_analysis_errors(
        self,
        *,
        device_id: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Devuelve errores con filename_hash (no filename real)."""
        clauses = []
        params: list = []
        if device_id:
            clauses.append('device_id = ?')
            params.append(device_id)
        if resolved is not None:
            clauses.append('resolved = ?')
            params.append(1 if resolved else 0)
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        params.append(int(limit))
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT id, timestamp, device_id, filename_hash, fingerprint, '
                '       error_class, error_msg, traceback, endpoint, '
                '       resolved, resolved_at '
                f'FROM analysis_errors{where} '
                'ORDER BY id DESC LIMIT ?',
                params,
            )
            return [dict(r) for r in c.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def get_errors_grouped(
        self,
        resolved: Optional[bool] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[Dict]:
        """Agrupa por (error_class, msg_short) con count y devices_affected.

        `since`/`until` filtran por rango de fechas. DEBEN venir en el mismo
        formato que la columna `timestamp` ('YYYY-MM-DD HH:MM:SS', UTC); si
        llega solo una fecha 'YYYY-MM-DD' se compara igual lexicograficamente.
        Cada grupo se enriquece con origin (cliente/servidor), platform,
        app_version y clean_msg via derive_error_meta — derivados del
        `endpoint`, que es la senal fiable, no del texto."""
        clauses = []
        params: list = []
        if resolved is not None:
            clauses.append('resolved = ?')
            params.append(1 if resolved else 0)
        if since:
            clauses.append('timestamp >= ?')
            params.append(since)
        if until:
            clauses.append('timestamp <= ?')
            params.append(until)
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT error_class, error_msg, device_id, timestamp, id, '
                '       resolved, traceback, filename_hash, endpoint '
                f'FROM analysis_errors{where} '
                'ORDER BY id DESC',
                params,
            )
            raw_rows = [dict(r) for r in c.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

        # Agrupacion en Python por (error_class, msg_normalizado). El normalizado
        # quita el prefijo '[plataforma version]' para fusionar el mismo bug a
        # traves de versiones (ver normalize_error_key). msg_short del grupo = la
        # clave normalizada, que es lo que el cliente devuelve para resolver.
        groups: Dict[tuple, dict] = {}
        for r in raw_rows:
            key = (r.get('error_class', ''), normalize_error_key(r.get('error_msg')))
            g = groups.get(key)
            ts = r.get('timestamp')
            if g is None:
                groups[key] = {
                    'error_class': key[0],
                    'msg_short': key[1],
                    'count': 1,
                    '_devices': {r.get('device_id')},
                    'first_seen': ts,
                    'last_seen': ts,
                    'latest_id': r.get('id'),
                    'unresolved_count': 1 if not r.get('resolved') else 0,
                    'sample_msg': r.get('error_msg'),
                    'sample_traceback': r.get('traceback'),
                    'sample_filename': r.get('filename_hash'),
                    'sample_endpoint': r.get('endpoint'),
                }
            else:
                g['count'] += 1
                g['_devices'].add(r.get('device_id'))
                if not r.get('resolved'):
                    g['unresolved_count'] += 1
                if ts and (g['first_seen'] is None or ts < g['first_seen']):
                    g['first_seen'] = ts
                if ts and (g['last_seen'] is None or ts > g['last_seen']):
                    g['last_seen'] = ts

        rows = []
        for g in groups.values():
            g['devices_affected'] = len(g.pop('_devices'))
            meta = derive_error_meta(
                g.get('error_class', ''),
                g.get('sample_msg'),
                g.get('sample_endpoint'),
            )
            g.update(meta)
            rows.append(g)

        rows.sort(key=lambda x: x['count'], reverse=True)
        return rows

    def count_errors(
        self,
        resolved: Optional[bool] = None,
        device_id: Optional[str] = None,
    ) -> int:
        """COUNT(*) de analysis_errors con filtros opcionales. Sustituye los
        `0` hardcodeados que /admin/stats devolvia antes de existir la tabla."""
        clauses = []
        params: list = []
        if resolved is not None:
            clauses.append('resolved = ?')
            params.append(1 if resolved else 0)
        if device_id:
            clauses.append('device_id = ?')
            params.append(device_id)
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        conn = self._open_conn()
        try:
            c = conn.cursor()
            row = c.execute(
                f'SELECT COUNT(*) AS n FROM analysis_errors{where}', params
            ).fetchone()
            return int(row['n']) if row else 0
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def prune_old_errors(self, days: int = 180, only_resolved: bool = True) -> int:
        """Borra errores viejos para que la tabla no crezca sin limite.

        Por defecto SOLO purga los ya resueltos (mas viejos que `days`),
        preservando el historial accionable. Devuelve filas borradas. No se
        ejecuta solo: se dispara desde el endpoint admin /errors/prune."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        clause = 'timestamp < ?'
        params: list = [cutoff]
        if only_resolved:
            clause += ' AND resolved = 1'
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(f'DELETE FROM analysis_errors WHERE {clause}', params)
            conn.commit()
            return c.rowcount
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def toggle_error_resolved(self, error_id: int) -> bool:
        """Toggle resolved flag. Devuelve el nuevo valor."""
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute('SELECT resolved FROM analysis_errors WHERE id = ?', (error_id,))
            row = c.fetchone()
            if not row:
                return False
            new_val = 0 if row['resolved'] else 1
            resolved_at = datetime.now(timezone.utc).isoformat() if new_val else None
            c.execute(
                'UPDATE analysis_errors SET resolved = ?, resolved_at = ? '
                'WHERE id = ?',
                (new_val, resolved_at, error_id),
            )
            conn.commit()
            return bool(new_val)
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()

    def resolve_error_group(self, error_class: str, msg_short: str) -> int:
        """Marca como resolved TODAS las occurrencias de un grupo, fusionando
        a traves de versiones. `msg_short` es la clave NORMALIZADA (sin prefijo
        '[plataforma version]') que devuelve get_errors_grouped. Resolvemos
        cualquier fila del mismo error_class cuyo mensaje normalice a esa clave,
        no importa la version -> los gemelos de '[ios 2.9.0]' y '[ios 2.9.2]'
        caen todos. Devuelve el numero de filas actualizadas.

        Si msg_short es '' (errores sin mensaje, p.ej. ClientDisconnect) se
        resuelven TODOS los errores sin resolver de ese error_class sin filtrar
        por mensaje — son todos del mismo bug logico."""
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                'SELECT id, error_msg FROM analysis_errors '
                'WHERE error_class = ? AND resolved = 0',
                (error_class,),
            )
            rows = c.fetchall()
            if msg_short:
                ids = [r['id'] for r in rows if normalize_error_key(r['error_msg']) == msg_short]
            else:
                # msg_short vacio: resolver todos los de este error_class cuyo
                # mensaje normalizado tambien sea vacio (mismo grupo sin mensaje).
                ids = [r['id'] for r in rows if not normalize_error_key(r['error_msg'])]
            if not ids:
                return 0
            resolved_at = datetime.now(timezone.utc).isoformat()
            c.executemany(
                'UPDATE analysis_errors SET resolved = 1, resolved_at = ? '
                'WHERE id = ?',
                [(resolved_at, _id) for _id in ids],
            )
            conn.commit()
            return len(ids)
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def count_engine_sources(self) -> Dict[str, int]:
        """Counts de tracks por engine_source.

        Los NULL (origen sin sellar) se reportan como 'unknown' EN VEZ de
        excluirse: ocultarlos hacia que el panel mostrara "todo render" cuando
        en realidad los analisis locales entran via /cache-analysis y se
        guardaban sin engine_source. Mostrarlos como 'unknown' refleja la
        realidad y deja ver como crece 'local_engine' segun se sellan."""
        conn = self._open_conn()
        try:
            c = conn.cursor()
            c.execute(
                "SELECT COALESCE(engine_source, 'unknown') AS engine_source, "
                'COUNT(*) AS n FROM tracks GROUP BY 1'
            )
            return {r['engine_source']: r['n'] for r in c.fetchall()}
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()

    def backfill_engine_source(self, value: str = 'local_engine',
                               dry_run: bool = True) -> Dict:
        """Sella engine_source en los tracks historicos sin etiquetar (NULL).

        Esos NULL vienen casi todos de /cache-analysis (subidas del motor
        local guardadas antes de instrumentar engine_source); los de Render ya
        estan sellados 'render'. dry_run=True solo inspecciona (cuantos NULL y
        su reparto por bpm_source, para verlo antes de tocar). dry_run=False
        aplica UPDATE engine_source=value WHERE engine_source IS NULL."""
        conn = self._open_conn()
        try:
            c = conn.cursor()
            total = c.execute('SELECT COUNT(*) AS n FROM tracks').fetchone()['n']
            null_before = c.execute(
                'SELECT COUNT(*) AS n FROM tracks WHERE engine_source IS NULL'
            ).fetchone()['n']
            # Reparto por bpm_source de los NULL — best-effort (requiere json1).
            # Util para ver cuantos son analisis reales ('local_engine') vs
            # imports/reconocimiento ('id3', 'rekordbox', 'audd', ...).
            bpm_breakdown: Dict[str, int] = {}
            try:
                rows = c.execute(
                    "SELECT COALESCE(json_extract(analysis_json,'$.bpm_source'),"
                    "'(sin bpm_source)') AS bs, COUNT(*) AS n "
                    'FROM tracks WHERE engine_source IS NULL GROUP BY bs '
                    'ORDER BY n DESC'
                ).fetchall()
                bpm_breakdown = {r['bs']: r['n'] for r in rows}
            except sqlite3.OperationalError:
                bpm_breakdown = {'(json_extract no disponible)': null_before}
            updated = 0
            if not dry_run and null_before:
                c.execute(
                    'UPDATE tracks SET engine_source = ? WHERE engine_source IS NULL',
                    (value,),
                )
                conn.commit()
                updated = c.rowcount
            return {
                'dry_run': dry_run,
                'value': value,
                'total_tracks': total,
                'null_before': null_before,
                'bpm_source_breakdown': bpm_breakdown,
                'updated': updated,
            }
        except sqlite3.OperationalError:
            return {'error': 'tabla tracks no disponible'}
        finally:
            conn.close()

    def get_fingerprint_stats(self) -> Dict[str, int]:
        """Stats agregadas para el panel admin: cuantos tracks tienen
        fingerprint, cuantos no, y cuantas colisiones hay (mismo fp,
        distinto id). Las colisiones son la senal mas clara de que el
        dedup actual exact-match esta dejando pasar duplicados — si la
        cifra crece con el tiempo, justifica invertir en Hamming distance
        (item 9 del backlog).
        """
        conn = self._open_conn()
        try:
            c = conn.cursor()
            stats = {
                'total_tracks': 0,
                'with_fingerprint': 0,
                'without_fingerprint': 0,
                'unique_fingerprints': 0,
                'collision_groups': 0,
                'collision_extra_rows': 0,
            }
            r = c.execute('SELECT COUNT(*) AS n FROM tracks').fetchone()
            stats['total_tracks'] = int(r['n']) if r else 0

            r = c.execute(
                'SELECT COUNT(*) AS n FROM tracks '
                "WHERE fingerprint IS NOT NULL AND fingerprint != ''"
            ).fetchone()
            stats['with_fingerprint'] = int(r['n']) if r else 0
            stats['without_fingerprint'] = stats['total_tracks'] - stats['with_fingerprint']

            r = c.execute(
                'SELECT COUNT(DISTINCT fingerprint) AS n FROM tracks '
                "WHERE fingerprint IS NOT NULL AND fingerprint != ''"
            ).fetchone()
            stats['unique_fingerprints'] = int(r['n']) if r else 0

            # Colisiones: grupos con >=2 tracks distintos compartiendo fp.
            # "collision_extra_rows" = filas que sobran (cada grupo de N
            # aporta N-1). Si suma 0, el dedup actual cubre todo.
            r = c.execute(
                'SELECT COUNT(*) AS groups, COALESCE(SUM(n - 1), 0) AS extras '
                'FROM (SELECT fingerprint, COUNT(*) AS n FROM tracks '
                "      WHERE fingerprint IS NOT NULL AND fingerprint != '' "
                '      GROUP BY fingerprint HAVING n > 1)'
            ).fetchone()
            if r:
                stats['collision_groups'] = int(r['groups'])
                stats['collision_extra_rows'] = int(r['extras'])
            return stats
        except sqlite3.OperationalError:
            return {
                'total_tracks': 0,
                'with_fingerprint': 0,
                'without_fingerprint': 0,
                'unique_fingerprints': 0,
                'collision_groups': 0,
                'collision_extra_rows': 0,
            }
        finally:
            conn.close()

    def count_analysis_sources(self) -> Dict[str, Dict[str, int]]:
        """Breakdown agregado de `*_source` (bpm, key, genre, track_type)
        a partir de los analisis guardados en `tracks.analysis_json`.

        Las columnas fisicas no existen — los sources viven dentro del
        JSON. Usamos `json_extract` (SQLite json1, presente en builds
        modernas). Si el motor no tiene json1, devuelve dict vacio.

        Returns:
            {
              'bpm': {'rekordbox': 12, 'local_engine': 80, ...},
              'key': {...},
              'genre': {...},
              'track_type': {...},
            }
        """
        fields = ('bpm_source', 'key_source', 'genre_source', 'track_type_source')
        out: Dict[str, Dict[str, int]] = {f.replace('_source', ''): {} for f in fields}
        conn = self._open_conn()
        try:
            c = conn.cursor()
            for field in fields:
                short = field.replace('_source', '')
                try:
                    rows = c.execute(
                        f"SELECT COALESCE(json_extract(analysis_json, '$.{field}'), 'unknown') AS src, "
                        '       COUNT(*) AS n '
                        '  FROM tracks '
                        ' WHERE analysis_json IS NOT NULL '
                        ' GROUP BY src'
                    ).fetchall()
                    for r in rows:
                        src = r['src'] or 'unknown'
                        out[short][str(src)] = int(r['n'])
                except sqlite3.OperationalError:
                    # json1 no disponible en este SQLite — saltar el campo.
                    continue
            return out
        finally:
            conn.close()

    def count_client_errors_by_context(self, since_hours: int = 24) -> Dict[str, int]:
        """Counts de errores cliente agrupados por context (sufijo despues
        de 'client:'). Util para el panel: ver de un vistazo si chromaprint
        / sync / analysis_api estan petando mas de lo normal en las ultimas
        24h. Tambien cuenta los 'unhandled:*' (middleware global) bajo la
        clave especial '_unhandled'.
        """
        from datetime import datetime, timedelta, timezone
        since_iso = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime('%Y-%m-%d %H:%M:%S')
        conn = self._open_conn()
        try:
            c = conn.cursor()
            rows = c.execute(
                'SELECT endpoint, COUNT(*) AS n FROM analysis_errors '
                'WHERE timestamp >= ? '
                "  AND (endpoint LIKE 'client:%' OR endpoint LIKE 'unhandled:%') "
                'GROUP BY endpoint',
                (since_iso,),
            ).fetchall()
            out: Dict[str, int] = {}
            for r in rows:
                ep = r['endpoint'] or ''
                if ep.startswith('client:'):
                    out[ep[len('client:'):]] = int(r['n'])
                elif ep.startswith('unhandled:'):
                    out['_unhandled'] = out.get('_unhandled', 0) + int(r['n'])
            return out
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()
