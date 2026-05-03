# Scripts one-shot del backend

Scripts utilitarios que NO se ejecutan automáticamente en el ciclo normal
de la app. Diseñados para correr a mano (LOCAL_ENGINE de developers o
shell remota en Render) cuando hace falta operación puntual.

## `migrate_to_chromaprint.py`

Re-procesa los `tracks` con `fingerprint_source = 'md5_legacy'` y
recalcula su fingerprint con Chromaprint cuando el archivo de audio sigue
accesible en disco.

### Cuándo usarlo

- Tras un upgrade a v2.8.0 sobre una BD existente con tracks pre-Chromaprint.
- Para sembrar la BD comunitaria (Render) con fingerprints Chromaprint
  reales antes de un release público — el dev ejecuta el script en su
  motor local (que tiene los archivos), los nuevos fingerprints se
  propagan a Render vía sync push.
- Periódicamente durante desarrollo para ver cuántos tracks legacy quedan.

### Uso

```bash
# Dry-run (default, no escribe nada). Ideal para ver el impacto.
python scripts/migrate_to_chromaprint.py --db /data/analysis.db

# Aplicar cambios — usa --apply.
python scripts/migrate_to_chromaprint.py --db /data/analysis.db --apply

# Local engine con BD junto al repo.
python scripts/migrate_to_chromaprint.py \
    --db local_analysis.db \
    --previews-dir ./previews_cache \
    --apply

# Limitar a N tracks (debug).
python scripts/migrate_to_chromaprint.py --db ... --apply --limit 50

# Buscar archivos por filename en directorios extra (cuando el path
# absoluto en analysis_json apunta a una máquina antigua).
python scripts/migrate_to_chromaprint.py --db ... --apply \
    --search-dir ~/Music --search-dir /Volumes/HDD/Tracks
```

### Qué hace al aplicar

Por cada track legacy con archivo accesible:

1. Calcula `chromaprint_helper.calculate_chromaprint_fingerprint(path)`.
2. Si el nuevo fp ≠ el viejo:
   - Verifica que no haya colisión con otro track en BD. Si la hay, salta.
   - `UPDATE tracks SET fingerprint=<nuevo>, id=<nuevo>, fingerprint_source='chromaprint'`.
   - `UPDATE corrections, dj_notes, community_cues, community_notes,
     beat_grid_corrections, track_popularity, track_ratings,
     audd_call_log` para apuntar al nuevo fp.
   - Renombra `previews_cache/<old>.mp3` → `previews_cache/<nuevo>.mp3`
     si existe.
3. Si el nuevo fp == el viejo (caso raro), solo marca `fingerprint_source='chromaprint'`.

### Qué NO hace

- No toca tracks sin archivo accesible (sincronizados desde otro device,
  paths foreign). Quedan con `fingerprint_source='md5_legacy'` esperando
  a que el dueño del archivo los reanalize o ejecute el script en su máquina.
- No borra previews ni metadata. Solo renombra cuando hace match exacto.
- No sobreescribe Chromaprint ya marcados — idempotente.

### Por qué dry-run por defecto

El script hace UPDATEs en cascada sobre 8 tablas y renombra archivos en
disco. Errores son muy difíciles de revertir sin backup. El dry-run
muestra qué se haría (`[DRY-RUN] Migraría <old> -> <new>`) sin tocar
nada — pasar `--apply` solo cuando estés seguro.

### Backup recomendado

Antes de `--apply` en una BD productiva:

```bash
sqlite3 /data/analysis.db ".backup /data/analysis.db.pre_chromaprint.bak"
```

Si algo va mal, restaurar:

```bash
cp /data/analysis.db.pre_chromaprint.bak /data/analysis.db
```

### Tests

`test_chromaprint.py` cubre `chromaprint_helper.calculate_chromaprint_fingerprint`
con mocks de `fpcalc`. La integración con la BD se valida con
`test_database_v2_8_0.py`. El script en sí no tiene tests automatizados;
correr el dry-run sobre una BD copia es la verificación recomendada.
