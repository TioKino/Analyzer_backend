# ==================== ENDPOINTS DE BÚSQUEDA ====================
# Añadir estos endpoints a main.py después de los endpoints existentes

from fastapi import Query

# --- Añadir después de /search-analyzed ---

@app.get("/search/artist/{artist}")
async def search_by_artist(artist: str, limit: int = Query(50, ge=1, le=200)):
    """Buscar tracks por artista"""
    results = db.search_by_artist(artist, limit)
    return {
        "query": artist,
        "count": len(results),
        "tracks": results
    }

@app.get("/search/genre/{genre}")
async def search_by_genre(genre: str, limit: int = Query(100, ge=1, le=500)):
    """Buscar tracks por género"""
    results = db.search_by_genre(genre, limit)
    return {
        "query": genre,
        "count": len(results),
        "tracks": results
    }

@app.get("/search/bpm")
async def search_by_bpm(
    min_bpm: float = Query(60, ge=30, le=300),
    max_bpm: float = Query(200, ge=30, le=300),
    limit: int = Query(100, ge=1, le=500)
):
    """Buscar tracks por rango de BPM"""
    results = db.search_by_bpm_range(min_bpm, max_bpm, limit)
    return {
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
        "count": len(results),
        "tracks": results
    }

@app.get("/search/energy")
async def search_by_energy(
    min_energy: int = Query(1, ge=1, le=10),
    max_energy: int = Query(10, ge=1, le=10),
    limit: int = Query(100, ge=1, le=500)
):
    """Buscar tracks por nivel de energía DJ (1-10)"""
    results = db.search_by_energy(min_energy, max_energy, limit)
    return {
        "min_energy": min_energy,
        "max_energy": max_energy,
        "count": len(results),
        "tracks": results
    }

@app.get("/search/key/{key}")
async def search_by_key(key: str, limit: int = Query(100, ge=1, le=500)):
    """Buscar tracks por tonalidad (ej: Am, C, 8A)"""
    results = db.search_by_key(key, limit)
    return {
        "key": key,
        "count": len(results),
        "tracks": results
    }

@app.get("/search/compatible/{camelot}")
async def search_compatible_keys(camelot: str, limit: int = Query(50, ge=1, le=200)):
    """
    Buscar tracks con tonalidades compatibles para mezcla.
    Usa reglas Camelot: mismo número, +1/-1, o cambio mayor/menor (A<->B)
    """
    results = db.search_compatible_keys(camelot, limit)
    
    # Calcular keys compatibles para mostrar en respuesta
    compatible_keys = []
    if camelot and len(camelot) >= 2:
        try:
            number = int(camelot[:-1])
            letter = camelot[-1].upper()
            prev_num = 12 if number == 1 else number - 1
            next_num = 1 if number == 12 else number + 1
            other_letter = 'B' if letter == 'A' else 'A'
            compatible_keys = [camelot, f'{prev_num}{letter}', f'{next_num}{letter}', f'{number}{other_letter}']
        except:
            pass
    
    return {
        "camelot": camelot,
        "compatible_keys": compatible_keys,
        "count": len(results),
        "tracks": results
    }

@app.get("/search/track-type/{track_type}")
async def search_by_track_type(track_type: str, limit: int = Query(100, ge=1, le=500)):
    """Buscar tracks por tipo (warmup, peak, closing)"""
    results = db.search_by_track_type(track_type, limit)
    return {
        "track_type": track_type,
        "count": len(results),
        "tracks": results
    }

# Modelo para búsqueda avanzada
class SearchRequest(BaseModel):
    artist: Optional[str] = None
    genre: Optional[str] = None
    min_bpm: Optional[float] = None
    max_bpm: Optional[float] = None
    min_energy: Optional[int] = None
    max_energy: Optional[int] = None
    key: Optional[str] = None
    track_type: Optional[str] = None
    limit: int = 100

@app.post("/search/advanced")
async def search_advanced(request: SearchRequest):
    """Búsqueda avanzada combinando múltiples criterios"""
    results = db.search_advanced(
        artist=request.artist,
        genre=request.genre,
        min_bpm=request.min_bpm,
        max_bpm=request.max_bpm,
        min_energy=request.min_energy,
        max_energy=request.max_energy,
        key=request.key,
        track_type=request.track_type,
        limit=request.limit
    )
    return {
        "filters": request.dict(exclude_none=True),
        "count": len(results),
        "tracks": results
    }

# ==================== ENDPOINTS DE BIBLIOTECA ====================

@app.get("/library/all")
async def get_all_tracks(limit: int = Query(1000, ge=1, le=5000)):
    """Obtener todos los tracks analizados"""
    results = db.get_all_tracks(limit)
    return {
        "count": len(results),
        "tracks": results
    }

@app.get("/library/artists")
async def get_unique_artists():
    """Obtener lista de artistas únicos"""
    artists = db.get_unique_artists()
    return {
        "count": len(artists),
        "artists": artists
    }

@app.get("/library/genres")
async def get_unique_genres():
    """Obtener lista de géneros únicos"""
    genres = db.get_unique_genres()
    return {
        "count": len(genres),
        "genres": genres
    }

@app.get("/library/stats")
async def get_library_stats():
    """Obtener estadísticas de la biblioteca"""
    return db.get_stats()

@app.get("/track/{track_id}")
async def get_track(track_id: str):
    """Obtener información de un track específico"""
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track no encontrado")
    return track

@app.delete("/track/{track_id}")
async def delete_track(track_id: str):
    """Eliminar un track de la base de datos"""
    deleted = db.delete_track(track_id)
    if not deleted:
        raise HTTPException(404, "Track no encontrado")
    return {"status": "ok", "message": "Track eliminado"}
