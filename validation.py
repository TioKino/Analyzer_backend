"""
DJ Analyzer Pro - Módulo de Validación
======================================
Validación de entrada para todos los endpoints.

Uso:
    from validation import (
        validate_audio_file,
        validate_bpm_range,
        validate_energy_range,
        sanitize_string,
        ValidationError
    )
"""

from fastapi import HTTPException, UploadFile
from typing import Optional, Tuple
import re
import os

# ============================================================================
# CONSTANTES DE VALIDACIÓN
# ============================================================================

# Límites de archivo
MAX_FILE_SIZE_MB = 100  # 100 MB máximo
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
ALLOWED_MIME_TYPES = {
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav',
    'audio/flac', 'audio/x-flac', 'audio/mp4', 'audio/m4a',
    'audio/aac', 'audio/ogg', 'application/octet-stream'
}

# Límites de BPM
MIN_BPM = 60.0
MAX_BPM = 200.0
DEFAULT_MIN_BPM = 70.0
DEFAULT_MAX_BPM = 180.0

# Límites de energía
MIN_ENERGY = 1
MAX_ENERGY = 10

# Límites de strings
MAX_STRING_LENGTH = 500
MAX_FILENAME_LENGTH = 255

# Límites de búsqueda
MAX_SEARCH_LIMIT = 1000
DEFAULT_SEARCH_LIMIT = 100

# Caracteres peligrosos
DANGEROUS_CHARS = re.compile(r'[<>"\';`\\]')
SQL_INJECTION_PATTERNS = re.compile(
    r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC)\b)',
    re.IGNORECASE
)

# Tonalidades válidas
VALID_KEYS = {
    'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 
    'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
    'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm',
    'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm'
}

VALID_CAMELOT = {
    '1A', '2A', '3A', '4A', '5A', '6A', '7A', '8A', '9A', '10A', '11A', '12A',
    '1B', '2B', '3B', '4B', '5B', '6B', '7B', '8B', '9B', '10B', '11B', '12B'
}

VALID_TRACK_TYPES = {'warmup', 'peak', 'closing', 'all'}


# ============================================================================
# EXCEPCIONES
# ============================================================================

class ValidationError(HTTPException):
    """Error de validación personalizado"""
    def __init__(self, detail: str, field: str = None):
        super().__init__(status_code=400, detail=detail)
        self.field = field


# ============================================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================================

async def validate_audio_file(
    file: UploadFile,
    max_size_mb: int = MAX_FILE_SIZE_MB,
    check_content: bool = True
) -> Tuple[bytes, str]:
    """
    Valida un archivo de audio subido.
    
    Args:
        file: Archivo subido
        max_size_mb: Tamaño máximo en MB
        check_content: Si debe verificar los magic bytes
    
    Returns:
        Tuple[bytes, str]: (contenido, extensión)
    
    Raises:
        ValidationError: Si el archivo no es válido
    """
    if not file or not file.filename:
        raise ValidationError("No se proporcionó archivo", "file")
    
    # Validar nombre de archivo
    filename = sanitize_filename(file.filename)
    if not filename:
        raise ValidationError("Nombre de archivo inválido", "filename")
    
    # Validar extensión
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Formato no soportado. Permitidos: {', '.join(ALLOWED_EXTENSIONS)}",
            "extension"
        )
    
    # Leer contenido
    content = await file.read()
    await file.seek(0)  # Reset para posible re-lectura
    
    # Validar tamaño
    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise ValidationError(
            f"Archivo demasiado grande. Máximo: {max_size_mb} MB",
            "size"
        )
    
    if len(content) < 1000:  # Mínimo 1KB
        raise ValidationError(
            "Archivo demasiado pequeño o corrupto",
            "size"
        )
    
    # Verificar magic bytes (opcional)
    if check_content:
        if not _is_valid_audio_content(content, ext):
            raise ValidationError(
                "El contenido no parece ser un archivo de audio válido",
                "content"
            )
    
    return content, ext


def _is_valid_audio_content(content: bytes, ext: str) -> bool:
    """Verifica magic bytes del archivo"""
    if len(content) < 12:
        return False
    
    # Magic bytes comunes
    magic_bytes = {
        '.mp3': [
            b'\xff\xfb',  # MP3 frame sync
            b'\xff\xfa',
            b'\xff\xf3',
            b'\xff\xf2',
            b'ID3',       # ID3 tag
        ],
        '.wav': [b'RIFF'],
        '.flac': [b'fLaC'],
        '.m4a': [b'ftyp', b'\x00\x00\x00'],
        '.aac': [b'\xff\xf1', b'\xff\xf9'],
        '.ogg': [b'OggS'],
    }
    
    expected = magic_bytes.get(ext, [])
    if not expected:
        return True  # No verificar si no tenemos magic bytes
    
    header = content[:12]
    return any(header.startswith(magic) or magic in header[:12] for magic in expected)


def sanitize_filename(filename: str) -> str:
    """Sanitiza nombre de archivo"""
    if not filename:
        return ""
    
    # Limitar longitud
    if len(filename) > MAX_FILENAME_LENGTH:
        ext = os.path.splitext(filename)[1]
        filename = filename[:MAX_FILENAME_LENGTH - len(ext)] + ext
    
    # Eliminar caracteres peligrosos de path
    filename = os.path.basename(filename)
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    return filename.strip()


# ============================================================================
# VALIDACIÓN DE RANGOS
# ============================================================================

def validate_bpm_range(
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None
) -> Tuple[float, float]:
    """
    Valida y normaliza rango de BPM.
    
    Returns:
        Tuple[float, float]: (min_bpm, max_bpm) validados
    """
    min_val = min_bpm if min_bpm is not None else DEFAULT_MIN_BPM
    max_val = max_bpm if max_bpm is not None else DEFAULT_MAX_BPM
    
    # Validar límites
    if min_val < MIN_BPM:
        min_val = MIN_BPM
    if max_val > MAX_BPM:
        max_val = MAX_BPM
    
    # Asegurar orden correcto
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    return min_val, max_val


def validate_energy_range(
    min_energy: Optional[int] = None,
    max_energy: Optional[int] = None
) -> Tuple[int, int]:
    """
    Valida y normaliza rango de energía.
    
    Returns:
        Tuple[int, int]: (min_energy, max_energy) validados
    """
    min_val = min_energy if min_energy is not None else MIN_ENERGY
    max_val = max_energy if max_energy is not None else MAX_ENERGY
    
    # Clamp a límites
    min_val = max(MIN_ENERGY, min(MAX_ENERGY, min_val))
    max_val = max(MIN_ENERGY, min(MAX_ENERGY, max_val))
    
    # Asegurar orden correcto
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    return min_val, max_val


def validate_limit(limit: int, max_limit: int = MAX_SEARCH_LIMIT) -> int:
    """Valida límite de búsqueda"""
    if limit < 1:
        return 1
    if limit > max_limit:
        return max_limit
    return limit


# ============================================================================
# VALIDACIÓN DE STRINGS
# ============================================================================

def sanitize_string(
    value: str,
    max_length: int = MAX_STRING_LENGTH,
    allow_empty: bool = True,
    field_name: str = "value"
) -> str:
    """
    Sanitiza un string de entrada.
    
    Args:
        value: String a sanitizar
        max_length: Longitud máxima permitida
        allow_empty: Si se permite string vacío
        field_name: Nombre del campo para errores
    
    Returns:
        String sanitizado
    
    Raises:
        ValidationError: Si el string no es válido
    """
    if value is None:
        if allow_empty:
            return ""
        raise ValidationError(f"{field_name} es requerido", field_name)
    
    if not isinstance(value, str):
        value = str(value)
    
    # Eliminar espacios extra
    value = ' '.join(value.split())
    
    # Validar longitud
    if len(value) > max_length:
        value = value[:max_length]
    
    if not value and not allow_empty:
        raise ValidationError(f"{field_name} no puede estar vacío", field_name)
    
    # Eliminar caracteres peligrosos
    value = DANGEROUS_CHARS.sub('', value)
    
    # Detectar posible SQL injection
    if SQL_INJECTION_PATTERNS.search(value):
        raise ValidationError(
            f"Caracteres no permitidos en {field_name}",
            field_name
        )
    
    return value


def validate_key(key: str) -> str:
    """Valida una tonalidad musical"""
    if not key:
        raise ValidationError("Tonalidad requerida", "key")
    
    key = key.strip()
    
    # Normalizar
    key = key.replace('♯', '#').replace('♭', 'b')
    key = key.replace(' minor', 'm').replace(' Minor', 'm')
    key = key.replace(' major', '').replace(' Major', '')
    
    if key not in VALID_KEYS:
        raise ValidationError(
            f"Tonalidad no válida: {key}. Ejemplos: Am, C, F#m",
            "key"
        )
    
    return key


def validate_camelot(camelot: str) -> str:
    """Valida una notación Camelot"""
    if not camelot:
        raise ValidationError("Camelot requerido", "camelot")
    
    camelot = camelot.strip().upper()
    
    if camelot not in VALID_CAMELOT:
        raise ValidationError(
            f"Camelot no válido: {camelot}. Ejemplos: 8A, 11B",
            "camelot"
        )
    
    return camelot


def validate_track_type(track_type: str) -> str:
    """Valida tipo de track"""
    if not track_type:
        return "all"
    
    track_type = track_type.strip().lower()
    
    if track_type not in VALID_TRACK_TYPES:
        raise ValidationError(
            f"Tipo no válido: {track_type}. Permitidos: {', '.join(VALID_TRACK_TYPES)}",
            "track_type"
        )
    
    return track_type


def validate_genre(genre: str) -> str:
    """Valida y sanitiza un género"""
    genre = sanitize_string(genre, max_length=100, field_name="genre")
    
    # Capitalizar primera letra de cada palabra
    if genre:
        genre = ' '.join(word.capitalize() for word in genre.split())
    
    return genre


# ============================================================================
# VALIDACIÓN DE IDs
# ============================================================================

def validate_track_id(track_id: str) -> str:
    """Valida un ID de track (fingerprint MD5)"""
    if not track_id:
        raise ValidationError("track_id requerido", "track_id")
    
    track_id = track_id.strip().lower()
    
    # MD5 tiene 32 caracteres hexadecimales
    if not re.match(r'^[a-f0-9]{32}$', track_id):
        # También permitir otros formatos de ID
        if not re.match(r'^[a-zA-Z0-9_-]{8,64}$', track_id):
            raise ValidationError(
                "ID de track no válido",
                "track_id"
            )
    
    return track_id


# ============================================================================
# MIDDLEWARE DE RATE LIMITING (Simple)
# ============================================================================

from collections import defaultdict
from datetime import datetime, timedelta
import threading

class SimpleRateLimiter:
    """
    Rate limiter simple en memoria.
    Para producción, usar Redis.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Verifica si el cliente puede hacer request"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        with self.lock:
            # Limpiar requests antiguos
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
            
            # Verificar límite
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False
            
            # Registrar request
            self.requests[client_id].append(now)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """Obtiene requests restantes"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        with self.lock:
            recent = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
            return max(0, self.requests_per_minute - len(recent))


# Instancia global del rate limiter
rate_limiter = SimpleRateLimiter(requests_per_minute=60)


def check_rate_limit(client_ip: str) -> None:
    """
    Verifica rate limit para un cliente.
    
    Raises:
        HTTPException: Si se excede el límite
    """
    if not rate_limiter.is_allowed(client_ip):
        remaining = rate_limiter.get_remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail="Demasiadas solicitudes. Intenta de nuevo en un minuto.",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)}
        )


# ============================================================================
# DECORADORES DE VALIDACIÓN
# ============================================================================

from functools import wraps
from fastapi import Request

def validate_audio_upload(max_size_mb: int = MAX_FILE_SIZE_MB):
    """Decorador para validar uploads de audio"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, file: UploadFile = None, **kwargs):
            if file:
                await validate_audio_file(file, max_size_mb)
            return await func(*args, file=file, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_client_ip(request: Request) -> str:
    """Obtiene IP del cliente (considerando proxies)"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
