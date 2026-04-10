"""
DJ ANALYZER - Configuración del Backend
============================================

Gestiona configuración usando variables de entorno para seguridad.

SETUP LOCAL:
    1. cp .env.example .env
    2. Edita .env con tus tokens
    3. python main.py

PRODUCCIÓN (Railway/Render):
    Configura las variables en el dashboard del hosting.
"""

import os
from pathlib import Path
from typing import Optional

# Intentar cargar python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_LOADED = True
except ImportError:
    _DOTENV_LOADED = False


# ==================== API TOKENS ====================

# AudD API - Identificación de canciones
# https://dashboard.audd.io/
AUDD_API_TOKEN: str = os.getenv('AUDD_API_TOKEN', '')

# Discogs API - Metadata de géneros
# https://www.discogs.com/settings/developers
DISCOGS_TOKEN: str = os.getenv('DISCOGS_TOKEN', '')
LASTFM_API_KEY: str = os.getenv('LASTFM_API_KEY', '')

# MusicBrainz User Agent (no requiere token)
MUSICBRAINZ_USER_AGENT: str = os.getenv(
    'MUSICBRAINZ_USER_AGENT',
    'DJAnalyzerPro/2.3.0 (https://github.com/tu-usuario/dj-analyzer-pro)'
)


# ==================== BASE DE DATOS ====================

DATABASE_PATH: str = os.getenv('DATABASE_PATH', 'analysis.db')
ARTWORK_CACHE_DIR: str = os.getenv('ARTWORK_CACHE_DIR', 'artwork_cache')
# On Render, use persistent disk to survive redeploys
_default_previews = '/data/previews' if os.getenv('RENDER') else 'previews_cache'
PREVIEWS_DIR: str = os.getenv('PREVIEWS_DIR', _default_previews)


# ==================== SERVIDOR ====================

# Host - 0.0.0.0 para aceptar conexiones externas
HOST: str = os.getenv('HOST', '0.0.0.0')

# Puerto - Railway/Render lo configuran automáticamente
PORT: int = int(os.getenv('PORT', '8000'))

# Modo debug
DEBUG: bool = os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes')

# URL base para generar enlaces de artwork
def _get_base_url() -> str:
    """Determina la URL base del servidor."""
    # 1. Variable de entorno explícita
    env_url = os.getenv('BASE_URL')
    if env_url:
        return env_url.rstrip('/')
    
    # 2. Railway proporciona RAILWAY_PUBLIC_DOMAIN
    railway_domain = os.getenv('RAILWAY_PUBLIC_DOMAIN')
    if railway_domain:
        return f'https://{railway_domain}'
    
    # 3. Render proporciona RENDER_EXTERNAL_URL
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    if render_url:
        return render_url.rstrip('/')
    
    # 4. Fallback a localhost
    return f'http://localhost:{PORT}'

BASE_URL: str = _get_base_url()


# ==================== RATE LIMITING ====================

RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() in ('true', '1')
RATE_LIMIT_REQUESTS: int = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '60'))  # segundos


# ==================== ANÁLISIS ====================

MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
ANALYSIS_TIMEOUT_SECONDS: int = int(os.getenv('ANALYSIS_TIMEOUT_SECONDS', '180'))
SUPPORTED_FORMATS: tuple = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma')


# ==================== SEGURIDAD ====================

# Secret compartido para HMAC-SHA256 en sync endpoints
# En producción: configurar en variables de entorno del hosting
SYNC_AUTH_SECRET: str = os.getenv('SYNC_AUTH_SECRET', '')

# Token para endpoints admin (reset-database, clear-cache)
ADMIN_TOKEN: str = os.getenv('ADMIN_TOKEN', '')


# ==================== CORS ====================

# Orígenes permitidos para CORS (separados por coma)
# En producción se recomienda configurar explícitamente
_cors_env = os.getenv('CORS_ORIGINS', '')
CORS_ORIGINS: list = _cors_env.split(',') if _cors_env else ['*']


# ==================== FUNCIONES DE UTILIDAD ====================

def validate_config() -> tuple[bool, list[str], list[str]]:
    """
    Valida la configuración.
    
    Returns:
        tuple: (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Tokens
    if not AUDD_API_TOKEN:
        warnings.append("AUDD_API_TOKEN no configurado - identificación deshabilitada")
    
    if not DISCOGS_TOKEN:
        warnings.append("DISCOGS_TOKEN no configurado - detección de género limitada")
    
    # Directorios
    try:
        Path(ARTWORK_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"No se pudo crear {ARTWORK_CACHE_DIR}: {e}")

    try:
        Path(PREVIEWS_DIR).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"No se pudo crear {PREVIEWS_DIR}: {e}")
    
    # Seguridad en producción
    is_production = bool(os.getenv('RENDER') or os.getenv('RAILWAY_ENVIRONMENT'))

    if not DEBUG and BASE_URL.startswith('http://localhost'):
        warnings.append("BASE_URL apunta a localhost en modo producción")

    if is_production and not SYNC_AUTH_SECRET:
        warnings.append("SYNC_AUTH_SECRET no configurado - sync sin autenticación")

    if is_production and not ADMIN_TOKEN:
        warnings.append("ADMIN_TOKEN no configurado - endpoints admin sin protección")

    if is_production and CORS_ORIGINS == ['*']:
        warnings.append("CORS_ORIGINS es wildcard (*) en producción - configurar orígenes explícitos")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def print_config():
    """Imprime la configuración actual (para debugging)."""
    is_valid, errors, warnings = validate_config()
    
    print("\n" + "=" * 55)
    print("  🎧 DJ ANALYZER - CONFIGURACIÓN")
    print("=" * 55)
    print(f"  📍 Modo:       {'DEBUG' if DEBUG else 'PRODUCCIÓN'}")
    print(f"  🌐 Base URL:   {BASE_URL}")
    print(f"  🔍 Host:Port:  {HOST}:{PORT}")
    print(f"  💾 Database:   {DATABASE_PATH}")
    print(f"  🖼️  Artwork:    {ARTWORK_CACHE_DIR}")
    print(f"  🔊 Previews:   {PREVIEWS_DIR}")
    print("-" * 55)
    print(f"  🔑 AudD:       {'✓ Configurado' if AUDD_API_TOKEN else '✗ No configurado'}")
    print(f"  🔑 Discogs:    {'✓ Configurado' if DISCOGS_TOKEN else '✗ No configurado'}")
    print(f"  📦 dotenv:     {'✓ Cargado' if _DOTENV_LOADED else '○ No disponible'}")
    
    if warnings:
        print("-" * 55)
        print("  ⚠️  ADVERTENCIAS:")
        for w in warnings:
            print(f"     • {w}")
    
    if errors:
        print("-" * 55)
        print("  ❌ ERRORES:")
        for e in errors:
            print(f"     • {e}")
    
    print("=" * 55 + "\n")
    return is_valid


def get_config_dict() -> dict:
    """Retorna configuración como diccionario (sin tokens)."""
    return {
        'version': '2.3.0',
        'debug': DEBUG,
        'base_url': BASE_URL,
        'database': DATABASE_PATH,
        'artwork_cache': ARTWORK_CACHE_DIR,
        'previews_cache': PREVIEWS_DIR,
        'audd_configured': bool(AUDD_API_TOKEN),
        'discogs_configured': bool(DISCOGS_TOKEN),
        'max_file_size_mb': MAX_FILE_SIZE_MB,
        'rate_limit_enabled': RATE_LIMIT_ENABLED,
    }


# ==================== INICIALIZACIÓN ====================

# Crear directorios necesarios al importar
Path(ARTWORK_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(PREVIEWS_DIR).mkdir(parents=True, exist_ok=True)
