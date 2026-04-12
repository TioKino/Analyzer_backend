"""
Route modules for DJ Analyzer Pro API.
Each module defines an APIRouter that main.py mounts via app.include_router().
"""
from .search import search_router, init as init_search
from .library import library_router, init as init_library
from .admin import admin_router, init as init_admin
from .community import community_router, init as init_community
from .preview import preview_router, init as init_preview
from .media import media_router, init as init_media
from .admin_panel import admin_panel_router


def init_all(database, previews_dir="", artwork_cache_dir="", generate_snippet_fn=None):
    """Initialize all route modules with shared dependencies."""
    init_search(database)
    init_library(database)
    init_admin(database)
    init_community(database)
    init_preview(database, previews_dir, generate_snippet_fn)
    init_media(database, artwork_cache_dir)
