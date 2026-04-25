"""
Bulk migration: sube todos los previews .mp3 existentes en el cache local
del engine PC al backend Render via /preview/upload/{id}.

Resuelve el problema de previews 404 en Mac/mobile para tracks analizados
en PC ANTES de que el push automatico estuviera wireado en /analyze
(main.py lineas 1849, 2062 ya hacen push para nuevos analisis).

Uso desde la carpeta del engine local en PC:
  python push_existing_previews.py
  python push_existing_previews.py --previews-dir .\\previews_cache --dry-run
  python push_existing_previews.py --skip-existing  (solo nuevos)

Idempotente: puede ejecutarse multiples veces sin riesgo. NO modifica los
.mp3 locales, solo los sube al disco persistente /data/previews/ de Render
via el endpoint POST /preview/upload/{id} (main.py linea 3362).

Validado: ejecutado el 2026-04-25, subio 37 previews en 30.9s, 0 fallos.
Resolvio los 404 que veia Mac al pedir previews de tracks analizados en PC.
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests no instalado. Instala con: pip install requests")
    sys.exit(1)


def push_preview(render_url, preview_path, retries=2):
    """Sube un preview a Render. Devuelve (ok, status_code)."""
    fingerprint = preview_path.stem  # nombre sin .mp3
    upload_url = f"{render_url.rstrip('/')}/preview/upload/{fingerprint}"

    for attempt in range(retries + 1):
        try:
            with open(preview_path, 'rb') as fp:
                files = {'file': (f'{fingerprint}.mp3', fp, 'audio/mpeg')}
                resp = requests.post(upload_url, files=files, timeout=20)
            if resp.status_code == 200:
                return True, 200
            if resp.status_code in (400, 413):
                # Errores de cliente: no reintentar (archivo invalido)
                return False, resp.status_code
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return False, resp.status_code
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            print(f"  [{fingerprint[:8]}] Error de red: {e}")
            return False, -1
    return False, -1


def main():
    parser = argparse.ArgumentParser(description="Bulk migration de previews PC -> Render")
    parser.add_argument(
        "--previews-dir",
        default=os.environ.get("PREVIEWS_DIR", "./previews_cache"),
        help="Directorio con los .mp3 (default: ./previews_cache o $PREVIEWS_DIR)",
    )
    parser.add_argument(
        "--render-url",
        default=os.environ.get("RENDER_SYNC_URL", "https://dj-analyzer-api.onrender.com"),
        help="URL base de Render (default: https://dj-analyzer-api.onrender.com)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo lista los archivos, no sube nada",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Antes de subir, hace check a Render. Si ya existe, salta.",
    )
    args = parser.parse_args()

    previews_dir = Path(args.previews_dir).resolve()
    if not previews_dir.is_dir():
        print(f"ERROR: directorio no existe: {previews_dir}")
        return 1

    mp3s = sorted(previews_dir.glob("*.mp3"))
    if not mp3s:
        print(f"No hay .mp3 en {previews_dir}")
        return 0

    print("=" * 70)
    print(f"Bulk push previews -> Render")
    print(f"  Origen:  {previews_dir}")
    print(f"  Destino: {args.render_url}/preview/upload/{{id}}")
    print(f"  Total:   {len(mp3s)} archivos")
    if args.dry_run:
        print(f"  Modo:    DRY-RUN (no sube nada)")
    if args.skip_existing:
        print(f"  Skip:    si ya existe en Render")
    print("=" * 70)

    if args.dry_run:
        for mp3 in mp3s:
            size_kb = mp3.stat().st_size // 1024
            print(f"  [{mp3.stem[:8]}] {mp3.name} ({size_kb}KB)")
        return 0

    ok = 0
    skipped = 0
    failed = []
    start_time = time.time()

    for i, mp3 in enumerate(mp3s, 1):
        fingerprint = mp3.stem

        if args.skip_existing:
            try:
                check = requests.get(
                    f"{args.render_url.rstrip('/')}/preview/{fingerprint}",
                    headers={'Range': 'bytes=0-0'},
                    timeout=5,
                )
                if check.status_code in (200, 206):
                    skipped += 1
                    if i % 20 == 0 or i == len(mp3s):
                        print(f"  [{i}/{len(mp3s)}] skip {fingerprint[:8]} (ya en Render)")
                    continue
            except requests.RequestException:
                pass  # Si el check falla, intentamos subir igual

        success, code = push_preview(args.render_url, mp3)
        if success:
            ok += 1
            if i % 10 == 0 or i == len(mp3s):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{len(mp3s)}] OK ({rate:.1f}/s, {ok} subidos, {skipped} skip, {len(failed)} fail)")
        else:
            failed.append((fingerprint, code))
            print(f"  [{i}/{len(mp3s)}] FAIL {fingerprint[:8]} -> HTTP {code}")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"DONE en {elapsed:.1f}s — subidos: {ok}, skip: {skipped}, fallidos: {len(failed)}")
    if failed:
        print("\nFallidos:")
        for fp, code in failed[:20]:
            print(f"  {fp[:16]}... HTTP {code}")
        if len(failed) > 20:
            print(f"  ... y {len(failed) - 20} mas")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
