# ============================================================================
# COMMUNITY CUES ENDPOINT - Zonas de cue validadas por la comunidad
# ============================================================================
# Aniadir a main.py despues de los endpoints existentes.
# Requiere aniadir la tabla community_cues en database.py
#
# Flujo:
#   1. DJ analiza track -> coloca cues manuales en CueFlow
#   2. Al exportar o sincronizar, envia sus cues al backend (POST /community-cues)
#   3. Backend agrega cues de todos los DJs por fingerprint
#   4. Cuando otro DJ analiza el mismo track, recibe zonas comunitarias (GET)
# ============================================================================

from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/community-cues", tags=["community"])


# ==================== MODELS ====================

class CueSubmission(BaseModel):
    """Un cue enviado por un DJ"""
    type: str  # mixIn, mixOut, drop, breakdown, vocal, buildup, loop, custom
    position_ms: int
    end_position_ms: Optional[int] = None  # para loops/regiones
    note: Optional[str] = None


class CueUpload(BaseModel):
    """Envio completo de cues de un DJ para un track"""
    fingerprint: str
    device_id: str  # identificador anonimo del dispositivo
    cues: List[CueSubmission]


class CommunityZoneResponse(BaseModel):
    type: str
    start: float  # segundos
    end: float  # segundos
    dj_count: int
    confidence: float
    top_note: Optional[str] = None


class CommunityResponse(BaseModel):
    fingerprint: str
    zones: List[CommunityZoneResponse]
    total_contributors: int
    last_updated: Optional[str] = None


# ==================== DATABASE ADDITIONS ====================
# Aniadir esto a AnalysisDB.init_db() en database.py:
#
# c.execute('''
#     CREATE TABLE IF NOT EXISTS community_cues (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         fingerprint TEXT NOT NULL,
#         device_id TEXT NOT NULL,
#         cue_type TEXT NOT NULL,
#         position_ms INTEGER NOT NULL,
#         end_position_ms INTEGER,
#         note TEXT,
#         created_at TEXT NOT NULL,
#         UNIQUE(fingerprint, device_id, cue_type, position_ms)
#     )
# ''')
# c.execute('CREATE INDEX IF NOT EXISTS idx_cc_fingerprint ON community_cues(fingerprint)')
# c.execute('CREATE INDEX IF NOT EXISTS idx_cc_device ON community_cues(device_id)')


# ==================== AGGREGATION LOGIC ====================

def aggregate_cues_into_zones(rows, duration_seconds: float = 0) -> List[dict]:
    """
    Agrega cues individuales de multiples DJs en zonas comunitarias.
    
    Algoritmo:
    1. Agrupa cues por tipo
    2. Para cada tipo, clustering por proximidad (ventana de 5 segundos)
    3. Para cada cluster: calcular centro, rango, conteo de DJs unicos, confianza
    4. Filtrar: minimo 2 DJs deben haber marcado la zona
    """
    from collections import defaultdict
    
    # Agrupar por tipo
    by_type = defaultdict(list)
    for row in rows:
        # row = (id, fingerprint, device_id, cue_type, position_ms, end_position_ms, note, created_at)
        by_type[row[3]].append({
            'device_id': row[2],
            'position_ms': row[4],
            'end_position_ms': row[5],
            'note': row[6],
        })
    
    zones = []
    cluster_window_ms = 5000  # 5 segundos de tolerancia
    
    for cue_type, cues in by_type.items():
        # Ordenar por posicion
        cues.sort(key=lambda c: c['position_ms'])
        
        # Clustering simple por proximidad
        clusters = []
        current_cluster = [cues[0]]
        
        for i in range(1, len(cues)):
            if cues[i]['position_ms'] - current_cluster[-1]['position_ms'] <= cluster_window_ms:
                current_cluster.append(cues[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [cues[i]]
        clusters.append(current_cluster)
        
        # Convertir clusters en zonas
        for cluster in clusters:
            unique_djs = set(c['device_id'] for c in cluster)
            dj_count = len(unique_djs)
            
            # Minimo 2 DJs para crear una zona comunitaria
            if dj_count < 2:
                continue
            
            positions = [c['position_ms'] for c in cluster]
            avg_pos = sum(positions) / len(positions)
            min_pos = min(positions)
            max_pos = max(positions)
            
            # Zona = rango de cues + margen de 2 segundos
            start_sec = max(0, (min_pos - 2000)) / 1000.0
            end_sec = (max_pos + 2000) / 1000.0
            if duration_seconds > 0:
                end_sec = min(end_sec, duration_seconds)
            
            # Confianza basada en acuerdo entre DJs
            # Si TODOS los DJs marcan el mismo punto, confianza alta
            spread_ms = max_pos - min_pos
            if spread_ms < 1000:
                confidence = 0.95
            elif spread_ms < 3000:
                confidence = 0.80
            elif spread_ms < 5000:
                confidence = 0.65
            else:
                confidence = 0.50
            
            # Boost confianza por cantidad de DJs
            confidence = min(1.0, confidence + (dj_count - 2) * 0.03)
            
            # Nota mas comun
            notes = [c['note'] for c in cluster if c.get('note')]
            top_note = max(set(notes), key=notes.count) if notes else None
            
            zones.append({
                'type': cue_type,
                'start': round(start_sec, 2),
                'end': round(end_sec, 2),
                'dj_count': dj_count,
                'confidence': round(confidence, 2),
                'top_note': top_note,
            })
    
    # Ordenar por posicion
    zones.sort(key=lambda z: z['start'])
    return zones


# ==================== ENDPOINTS ====================

def register_community_endpoints(app, db):
    """Registra los endpoints de community cues en la app FastAPI"""
    
    @app.post("/community-cues", response_model=dict)
    async def upload_community_cues(upload: CueUpload):
        """
        Un DJ sube sus cues para un track (por fingerprint).
        Se borran cues anteriores del mismo device_id para ese fingerprint
        y se insertan los nuevos.
        """
        from datetime import datetime
        
        if not upload.fingerprint or not upload.cues:
            return {"status": "error", "message": "fingerprint y cues requeridos"}
        
        conn = db.conn
        c = conn.cursor()
        
        try:
            # Borrar cues anteriores de este device para este track
            c.execute(
                'DELETE FROM community_cues WHERE fingerprint = ? AND device_id = ?',
                (upload.fingerprint, upload.device_id)
            )
            
            # Insertar nuevos cues
            now = datetime.now().isoformat()
            for cue in upload.cues:
                c.execute('''
                    INSERT INTO community_cues 
                    (fingerprint, device_id, cue_type, position_ms, end_position_ms, note, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    upload.fingerprint,
                    upload.device_id,
                    cue.type,
                    cue.position_ms,
                    cue.end_position_ms,
                    cue.note,
                    now,
                ))
            
            conn.commit()
            
            # Devolver zonas actualizadas
            c.execute(
                'SELECT * FROM community_cues WHERE fingerprint = ?',
                (upload.fingerprint,)
            )
            all_rows = c.fetchall()
            unique_devices = set(r[2] for r in all_rows)
            
            # Obtener duracion del track si existe
            c.execute('SELECT duration FROM tracks WHERE fingerprint = ?', (upload.fingerprint,))
            dur_row = c.fetchone()
            duration = dur_row[0] if dur_row else 0
            
            zones = aggregate_cues_into_zones(all_rows, duration)
            
            return {
                "status": "ok",
                "cues_saved": len(upload.cues),
                "total_contributors": len(unique_devices),
                "zones": zones,
            }
            
        except Exception as e:
            logger.error(f"Error saving community cues: {e}")
            return {"status": "error", "message": str(e)}
    
    @app.get("/community-cues/{fingerprint}", response_model=CommunityResponse)
    async def get_community_cues(fingerprint: str):
        """
        Obtiene las zonas comunitarias agregadas para un track.
        Solo devuelve zonas con >= 2 DJs de acuerdo.
        """
        conn = db.conn
        c = conn.cursor()
        
        c.execute(
            'SELECT * FROM community_cues WHERE fingerprint = ?',
            (fingerprint,)
        )
        rows = c.fetchall()
        
        if not rows:
            return CommunityResponse(
                fingerprint=fingerprint,
                zones=[],
                total_contributors=0,
            )
        
        unique_devices = set(r[2] for r in rows)
        
        # Obtener duracion
        c.execute('SELECT duration FROM tracks WHERE fingerprint = ?', (fingerprint,))
        dur_row = c.fetchone()
        duration = dur_row[0] if dur_row else 0
        
        zones_raw = aggregate_cues_into_zones(rows, duration)
        zones = [CommunityZoneResponse(**z) for z in zones_raw]
        
        # Ultima actualizacion
        last = max(r[7] for r in rows) if rows else None
        
        return CommunityResponse(
            fingerprint=fingerprint,
            zones=zones,
            total_contributors=len(unique_devices),
            last_updated=last,
        )
    
    @app.get("/community-cues/{fingerprint}/raw")
    async def get_raw_community_cues(fingerprint: str):
        """
        Devuelve todos los cues individuales (sin agregar) para debug.
        """
        conn = db.conn
        c = conn.cursor()
        c.execute(
            'SELECT cue_type, position_ms, end_position_ms, note, device_id, created_at '
            'FROM community_cues WHERE fingerprint = ? ORDER BY position_ms',
            (fingerprint,)
        )
        rows = c.fetchall()
        return {
            "fingerprint": fingerprint,
            "total_cues": len(rows),
            "cues": [
                {
                    "type": r[0],
                    "position_ms": r[1],
                    "end_position_ms": r[2],
                    "note": r[3],
                    "device_id": r[4][:8] + "...",  # anonimizar
                    "created_at": r[5],
                }
                for r in rows
            ],
        }
    
    logger.info("Community cues endpoints registered")
