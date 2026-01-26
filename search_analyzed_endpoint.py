# ==================== ENDPOINT SEARCH-ANALYZED ====================
# Añadir este código en main.py, después de los otros endpoints de búsqueda
# (después de la línea con @app.get("/search/compatible/{camelot}") aproximadamente)

# CÓDIGO A AÑADIR EN main.py:

@app.get("/search-analyzed")
async def search_analyzed_track(
    artist: str = Query(..., description="Nombre del artista"),
    title: str = Query(..., description="Título del track")
):
    """
    Busca si un track ya fue analizado por algún usuario.
    Devuelve TODA la información del análisis si existe.
    
    Returns:
        - found: bool - Si se encontró el track
        - track: dict - Toda la información del análisis (si existe)
        - in_collective: bool - Si está en la memoria colectiva
    """
    import re
    
    # Validar y sanitizar entrada
    artist_clean = sanitize_string(artist, max_length=200, allow_empty=False, field_name="artist")
    title_clean = sanitize_string(title, max_length=200, allow_empty=False, field_name="title")
    
    # Normalizar para búsqueda
    artist_normalized = artist_clean.lower().strip()
    title_normalized = re.sub(
        r'\s*\(?(Original Mix|Extended Mix|Radio Edit|Remix|Club Mix|Dub Mix)\)?', 
        '', 
        title_clean, 
        flags=re.IGNORECASE
    ).lower().strip()
    
    try:
        conn = db.conn
        cursor = conn.cursor()
        
        # Búsqueda exacta primero
        cursor.execute("""
            SELECT * FROM tracks 
            WHERE LOWER(artist) = ? AND LOWER(title) LIKE ?
            AND bpm IS NOT NULL AND bpm > 0
            ORDER BY analyzed_at DESC
            LIMIT 1
        """, (artist_normalized, f"%{title_normalized}%"))
        
        row = cursor.fetchone()
        
        if not row:
            # Búsqueda más flexible
            cursor.execute("""
                SELECT * FROM tracks 
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                AND bpm IS NOT NULL AND bpm > 0
                ORDER BY analyzed_at DESC
                LIMIT 1
            """, (f"%{artist_normalized}%", f"%{title_normalized}%"))
            row = cursor.fetchone()
        
        if row:
            # Convertir a dict usando el método existente
            track_dict = db._row_to_dict(row)
            
            # Si hay analysis_json, parsear para obtener todos los campos
            if track_dict and track_dict.get('analysis_json'):
                try:
                    full_analysis = json.loads(track_dict['analysis_json'])
                    # Combinar con los campos básicos
                    track_dict.update(full_analysis)
                except:
                    pass
            
            # Eliminar el JSON crudo del response
            if track_dict and 'analysis_json' in track_dict:
                del track_dict['analysis_json']
            
            return {
                "found": True,
                "in_collective": True,
                "track": track_dict
            }
        
        return {
            "found": False,
            "in_collective": False,
            "track": None
        }
        
    except Exception as e:
        print(f"Error en search-analyzed: {e}")
        return {
            "found": False,
            "in_collective": False,
            "track": None,
            "error": str(e)
        }
