#!/usr/bin/env python3
"""
Valida la MEMORIA COLECTIVA POR SONIDO con audio REAL (sin servidor ni BD).

Computa la huella acustica Chromaprint (fpcalc) de cada archivo que le pases,
mide la distancia Hamming entre cada par y dice si comparten CLUSTER acustico
(es decir, si el backend los agruparia como el mismo audio -> misma memoria
colectiva: cues, beat-grid, votos, ratings).

Uso:
    python3 scripts/validate_acoustic.py cancion.flac cancion.mp3
    python3 scripts/validate_acoustic.py cancion.flac cancion.mp3 otra_distinta.mp3

Requisitos:
    - fpcalc en el PATH.  macOS:  brew install chromaprint
                          Linux:  apt install libchromaprint-tools

Interpretacion:
    - Mismo tema en flac y mp3  -> distancia ~0.00-0.02 -> MISMO cluster  (OK)
    - Dos temas distintos       -> distancia ~0.40-0.50 -> clusters DISTINTOS (OK)
"""
import os
import sys

# Permite importar acoustic_fingerprint desde la raiz del repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acoustic_fingerprint import (  # noqa: E402
    MATCH_THRESHOLD,
    compute_raw_chromaprint,
    fingerprints_match,
    hamming_distance,
)


def main():
    files = sys.argv[1:]
    if len(files) < 2:
        print(__doc__)
        print("ERROR: pasa al menos DOS archivos (mismo tema en 2 formatos).")
        sys.exit(1)

    prints = {}
    for f in files:
        if not os.path.exists(f):
            print(f"[X] No existe: {f}")
            sys.exit(1)
        raw = compute_raw_chromaprint(f)
        if not raw:
            print(f"[X] fpcalc no pudo con: {f}")
            print("    Instala fpcalc:  macOS -> brew install chromaprint")
            print("                     Linux -> apt install libchromaprint-tools")
            sys.exit(1)
        prints[f] = raw
        print(f"[OK] {os.path.basename(f)}: {len(raw)} subfingerprints")

    print(f"\nUmbral de match: {MATCH_THRESHOLD} (por debajo = MISMO audio)\n")

    names = list(prints)
    all_ok = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = hamming_distance(prints[a], prints[b])
            match = fingerprints_match(prints[a], prints[b])
            verdict = ("MISMO cluster -> COMPARTEN memoria colectiva"
                       if match else
                       "clusters DISTINTOS -> memoria separada")
            print(f"{os.path.basename(a)}  vs  {os.path.basename(b)}")
            print(f"   Hamming = {d:.4f}   =>   {verdict}\n")

    print("Recordatorio: un flac + su mp3 del MISMO tema deben salir MISMO "
          "cluster.\nDos temas DISTINTOS deben salir clusters DISTINTOS.")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
