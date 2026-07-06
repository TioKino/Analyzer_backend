"""
============================================================
CLASIFICACIÓN DE GÉNERO POR ANÁLISIS ESPECTRAL
============================================================

Sistema UNIVERSAL de clasificación de género basado en características de audio
(BPM, energía, espectro, etc.). Se usa como FALLBACK de MÍNIMA prioridad: lo
pisan Discogs, MusicBrainz, ID3 y el consenso comunitario.

## De cascada de `return` a SCORING (2026-07-06)

El diseño anterior era una cascada: `if 120 <= bpm <= 140: ... return "Techno"`.
Cada banda terminaba en un `return` INCONDICIONAL, así que el primer rango de BPM
que matcheaba decidía y tapaba a todos los géneros de rangos SOLAPADOS. Resultado:
~17 géneros eran CÓDIGO MUERTO (un Reggae a 75 BPM salía "Hip Hop"; un Ambient,
un Disco, un Dubstep a 140, un Jungle... jamás se alcanzaban en su rango).

Ahora es SCORING sobre perfiles declarativos:
  - Un perfil `(genero, bpm_lo, bpm_hi, [condiciones])` es CANDIDATO si el BPM
    cae en su rango y se cumplen TODAS sus condiciones.
  - Score = nº de condiciones cumplidas. El género MÁS ESPECÍFICO (más
    condiciones) gana al fallback genérico de su familia. Empate -> el PRIMERO
    en la lista (por eso el ORDEN de `_GENRE_PROFILES` es la "prioridad de
    género": la decisión de producto sobre qué género posee cada zona de BPM
    cuando las señales son neutras).
  - Si ningún perfil matchea, cae al fallback por energía.

Ningún género queda inalcanzable: si sus condiciones se cumplen en su rango de
BPM, es candidato — haya o no otros géneros solapando ese rango.
"""

import numpy as np


def _extract_features(bpm, energy, has_bass, percussion_density,
                      spectral_centroid, rolloff):
    """Deriva las señales que consumen los perfiles. Mismos umbrales que la
    cascada original (is_melodic/is_dark/is_bright) para no mover los casos
    core."""
    brightness = float(np.mean(spectral_centroid))
    spectral_std = float(np.std(spectral_centroid))
    return {
        'bpm': bpm,
        'energy': energy,
        'has_bass': bool(has_bass),
        'perc': percussion_density,
        'brightness': brightness,
        'rolloff': float(np.mean(rolloff)),
        'spectral_std': spectral_std,
        'is_melodic': spectral_std > 800,   # alta variación espectral
        'is_dark': brightness < 2000,
        'is_bright': brightness > 3500,
    }


# ============================================================
# PERFILES DE GÉNERO
# ============================================================
# (genero, bpm_lo, bpm_hi, [condiciones]). Candidato si bpm en [lo, hi] y TODAS
# las condiciones (lambdas sobre el dict de features) se cumplen. El orden marca
# la prioridad en empates. Traducido de la cascada original preservando umbrales.
#
# NOTA de rangos: las familias tienen un fallback de rango "propio" (Hip Hop
# 60-100, House 115-128, Techno 128-140, Trance 140-150, Hard Dance 150-160, DnB
# 160-185). Los sub-géneros que viven en tempos de otra familia (Dubstep a 140,
# Disco a 120, Jungle a 165...) llevan condiciones, así que ganan por score
# cuando sus señales encajan. DnB va ANTES que Hard Dance para que 160 exacto
# caiga en DnB (tradeoff documentado: Gabber/Speedcore 170+ caen como DnB).

_GENRE_PROFILES = [
    # ── Hip Hop / Trap / R&B / Lo-Fi / Phonk (60-100) ──
    ('Trap',            60, 100, [lambda f: f['perc'] > 0.6, lambda f: f['has_bass'], lambda f: f['brightness'] > 2500]),
    ('Phonk',           60, 100, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6]),
    ('Hip Hop',         60, 100, [lambda f: f['perc'] > 0.6, lambda f: f['has_bass']]),
    ('R&B',             60, 100, [lambda f: f['is_melodic'], lambda f: f['energy'] < 0.5]),
    ('Lo-Fi Hip Hop',   60, 100, [lambda f: f['energy'] < 0.4]),
    ('Hip Hop',         60, 100, []),  # fallback de familia

    # ── Reggae / Dub (60-90) ──
    ('Dub',             60, 90,  [lambda f: f['has_bass'], lambda f: f['energy'] < 0.6, lambda f: f['is_dark']]),
    ('Reggae',          60, 90,  [lambda f: f['has_bass'], lambda f: f['energy'] < 0.6]),

    # ── Ambient / Downtempo (55-100, energía baja) ──
    ('Dark Ambient',    55, 100, [lambda f: f['energy'] < 0.4, lambda f: f['perc'] < 0.2, lambda f: f['is_dark']]),
    ('Ambient',         55, 100, [lambda f: f['energy'] < 0.4, lambda f: f['perc'] < 0.2]),
    ('Downtempo',       55, 100, [lambda f: f['energy'] < 0.4, lambda f: f['is_melodic']]),
    ('Chillout',        55, 100, [lambda f: f['energy'] < 0.4]),

    # ── Synthwave (80-120) ──
    ('Darksynth',       80, 120, [lambda f: f['brightness'] > 3000, lambda f: f['is_melodic'], lambda f: f['is_dark']]),
    ('Synthwave',       80, 120, [lambda f: f['brightness'] > 3000, lambda f: f['is_melodic']]),

    # ── Latin / Reggaeton (100-115) ──
    ('Reggaeton',       100, 115, [lambda f: f['perc'] > 0.7, lambda f: f['has_bass'], lambda f: f['energy'] > 0.6]),
    ('Latin Urban',     100, 115, [lambda f: f['energy'] > 0.7, lambda f: f['has_bass']]),
    ('Dembow',          100, 115, [lambda f: f['perc'] > 0.7]),
    ('Bachata',         100, 115, [lambda f: f['is_melodic'], lambda f: f['energy'] < 0.5]),
    ('Latin',           100, 115, []),  # fallback de familia

    # ── Moombahton (100-112) ──
    ('Moombahton',      100, 112, [lambda f: f['has_bass'], lambda f: f['perc'] > 0.6]),

    # ── Pop / Dance Pop / EDM (100-130) ──
    ('Dance Pop',       100, 130, [lambda f: f['is_melodic'], lambda f: f['is_bright'], lambda f: f['energy'] > 0.6]),
    ('EDM',             100, 130, [lambda f: f['energy'] > 0.7, lambda f: f['brightness'] > 3000]),
    ('Pop',             100, 130, [lambda f: f['is_melodic'], lambda f: f['is_bright']]),

    # ── Disco / Funk (110-130) ──
    ('Disco',           110, 130, [lambda f: f['is_melodic'], lambda f: f['brightness'] > 2500, lambda f: f['energy'] > 0.6]),
    ('Nu Disco',        110, 130, [lambda f: f['is_melodic'], lambda f: f['brightness'] > 2500]),
    ('Funk',            110, 130, [lambda f: f['perc'] > 0.5, lambda f: f['has_bass']]),

    # ── House (115-128) ──
    ('Tech House',      115, 128, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.7, lambda f: f['perc'] > 0.7]),
    ('Bass House',      115, 128, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.7, lambda f: f['is_bright']]),
    ('Progressive House', 115, 128, [lambda f: f['brightness'] > 3000, lambda f: f['is_melodic']]),
    ('Electro House',   115, 128, [lambda f: f['brightness'] > 3000]),
    ('Deep House',      115, 128, [lambda f: f['is_dark'], lambda f: f['energy'] < 0.5]),
    ('Organic House',   115, 128, [lambda f: f['is_melodic'], lambda f: f['energy'] < 0.6, lambda f: f['brightness'] < 2500]),
    ('Melodic House',   115, 128, [lambda f: f['is_melodic'], lambda f: f['energy'] < 0.6]),
    ('Minimal House',   115, 128, [lambda f: f['perc'] < 0.4]),
    ('Jackin House',    115, 128, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.5]),
    ('House',           115, 128, []),  # fallback de familia

    # ── Electro (125-135) ──
    ('Electro',         125, 135, [lambda f: f['brightness'] > 3000, lambda f: f['perc'] > 0.5]),

    # ── Big Room / Festival (126-132) ──
    ('Big Room',        126, 132, [lambda f: f['energy'] > 0.8, lambda f: f['brightness'] > 3500]),
    ('Festival Progressive', 126, 132, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.7]),

    # ── Techno (128-140) ──
    ('Industrial Techno', 128, 140, [lambda f: f['energy'] > 0.75, lambda f: f['perc'] > 0.7, lambda f: f['is_dark']]),
    ('Hard Techno',     128, 140, [lambda f: f['energy'] > 0.75, lambda f: f['perc'] > 0.7]),
    ('Peak Time Techno', 128, 140, [lambda f: f['energy'] > 0.65, lambda f: 0.5 < f['perc'] < 0.8]),
    ('Acid Techno',     128, 140, [lambda f: f['brightness'] > 3500, lambda f: f['energy'] > 0.55]),
    ('Detroit Techno',  128, 140, [lambda f: 0.4 < f['perc'] < 0.7, lambda f: f['brightness'] > 2500]),
    ('Melodic Techno',  128, 140, [lambda f: f['is_melodic'], lambda f: f['brightness'] > 2800, lambda f: 0.4 < f['energy'] < 0.7]),
    ('Dub Techno',      128, 140, [lambda f: f['is_dark'], lambda f: f['energy'] < 0.5, lambda f: f['perc'] < 0.5]),
    ('Hypnotic Techno', 128, 140, [lambda f: f['energy'] < 0.5, lambda f: f['perc'] < 0.5]),
    ('Minimal Techno',  128, 140, [lambda f: f['perc'] < 0.4]),
    ('Berlin Techno',   128, 140, [lambda f: f['has_bass'], lambda f: f['is_dark'], lambda f: f['energy'] > 0.5]),
    ('Techno',          128, 140, []),  # fallback de familia

    # ── Dubstep half-time (136-145) — antes de Trance/Grime por especificidad ──
    ('Brostep',         136, 145, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6, lambda f: f['brightness'] > 3000]),
    ('Deep Dubstep',    136, 145, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6, lambda f: f['is_dark']]),
    ('Riddim',          136, 145, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6, lambda f: f['perc'] > 0.7]),
    ('Dubstep',         136, 145, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6]),
    ('Melodic Dubstep', 136, 145, [lambda f: f['is_melodic'], lambda f: f['brightness'] > 3000]),

    # ── Grime (138-142) ──
    ('Grime',           138, 142, [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6, lambda f: f['brightness'] < 2500]),

    # ── Breakbeat / UK Garage (130-145) ──
    ('Breakbeat',       130, 145, [lambda f: f['spectral_std'] > 1000, lambda f: f['perc'] > 0.6, lambda f: f['brightness'] > 2500]),
    ('UK Garage',       130, 145, [lambda f: f['spectral_std'] > 1000, lambda f: f['perc'] > 0.6]),

    # ── Trance (140-150) + Psytrance ──
    ('Uplifting Trance', 140, 150, [lambda f: f['is_bright'], lambda f: f['is_melodic'], lambda f: f['energy'] > 0.7]),
    ('Vocal Trance',    140, 150, [lambda f: f['is_bright'], lambda f: f['is_melodic']]),
    ('Progressive Trance', 140, 150, [lambda f: f['energy'] > 0.65, lambda f: f['brightness'] > 2500]),
    ('Tech Trance',     140, 150, [lambda f: f['is_dark'], lambda f: f['energy'] > 0.7]),
    ('Psytrance',       145, 150, [lambda f: f['energy'] > 0.6]),
    ('Trance',          140, 150, []),  # fallback de familia

    # ── Future Bass (140-160) ──
    ('Future Bass',     140, 160, [lambda f: f['is_bright'], lambda f: f['is_melodic'], lambda f: f['energy'] > 0.5]),

    # ── Dubstep 70 BPM (68-75) ──
    ('Brostep',         68, 75,  [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6, lambda f: f['brightness'] > 3000]),
    ('Deep Dubstep',    68, 75,  [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6, lambda f: f['is_dark']]),
    ('Dubstep',         68, 75,  [lambda f: f['has_bass'], lambda f: f['energy'] > 0.6]),
    ('Chillstep',       68, 75,  [lambda f: f['energy'] < 0.4]),

    # ── Drum & Bass (160-185) — ANTES que Hard Dance para poseer 160 ──
    ('Liquid Drum & Bass', 160, 185, [lambda f: f['brightness'] > 3000, lambda f: f['is_melodic']]),
    ('Neurofunk',       160, 185, [lambda f: f['is_dark'], lambda f: f['energy'] > 0.7]),
    ('Jump Up',         160, 185, [lambda f: f['perc'] > 0.8]),
    ('Atmospheric DnB', 160, 185, [lambda f: f['is_dark'], lambda f: f['energy'] < 0.5]),
    ('Drum & Bass',     160, 185, []),  # fallback de familia

    # ── Jungle (150-175, breakbeats) ──
    ('Jungle',          150, 175, [lambda f: f['perc'] > 0.75, lambda f: f['spectral_std'] > 1000]),

    # ── Hard Dance (150-160) — Hardstyle/Frenchcore ──
    ('Euphoric Hardstyle', 150, 160, [lambda f: f['energy'] > 0.8, lambda f: f['is_melodic']]),
    ('Rawstyle',        150, 160, [lambda f: f['energy'] > 0.8]),
    ('Frenchcore',      150, 160, [lambda f: f['is_dark'], lambda f: f['has_bass']]),
    ('Hardstyle',       150, 160, []),  # fallback de familia

    # ── Rock / Metal (80-200) ──
    ('Thrash Metal',    160, 200, [lambda f: f['brightness'] > 4000, lambda f: f['energy'] > 0.8]),
    ('Heavy Metal',     140, 200, [lambda f: f['brightness'] > 4000, lambda f: f['energy'] > 0.8]),
    ('Hard Rock',       80, 200,  [lambda f: f['brightness'] > 4000, lambda f: f['energy'] > 0.8]),
    ('Rock',            100, 140, [lambda f: f['is_melodic'], lambda f: f['energy'] > 0.6, lambda f: f['brightness'] > 3800]),
    ('Alternative Rock', 100, 140, [lambda f: f['is_melodic'], lambda f: f['brightness'] > 3800]),

    # ── Jazz (variable) ──
    # Restrictivo A PROPOSITO: en una app de DJ el jazz es raro y sus señales
    # espectrales (melodico, poca percusion, sin bajo) las comparte MUCHA
    # electronica melodica. Sin acotar, sus 4 condiciones lo hacian ganar sobre
    # Synthwave/Pop/Melodic House. Lo distinguimos por percusion MUY escasa
    # (instrumentacion en vivo, no programada) + timbre acustico de rango medio
    # (ni synth brillante ni sub pesado).
    ('Jazz',            60, 200,  [lambda f: f['perc'] < 0.3, lambda f: not f['has_bass'], lambda f: f['spectral_std'] > 1200, lambda f: 1800 < f['brightness'] < 3000]),
]


def _energy_fallback(f):
    """Último recurso cuando ningún perfil matchea: por nivel de energía."""
    if f['energy'] > 0.7:
        return 'Hard Dance' if f['bpm'] > 140 else 'EDM'
    if f['energy'] < 0.4:
        return 'Chillout' if f['is_melodic'] else 'Ambient'
    return 'Electronic'


def classify_genre_advanced(bpm: float, energy: float, has_bass: bool,
                            y, sr, percussion_density: float,
                            spectral_centroid, rolloff) -> str:
    """
    Clasifica el género musical por análisis espectral (fuente de MÍNIMA
    prioridad; la pisan Discogs/MusicBrainz/ID3/consenso).

    Parámetros (firma estable — la llama main.py:analyze):
    - bpm: tempo detectado
    - energy: energía normalizada (0-1)
    - has_bass: graves prominentes
    - y, sr: señal/sample-rate (no usados hoy; se conservan por compat)
    - percussion_density: densidad de percusión (0-1)
    - spectral_centroid, rolloff: arrays espectrales

    Devuelve el género con más señales confirmadas en su rango de BPM (ver
    docstring del módulo). Nunca deja un género inalcanzable.
    """
    f = _extract_features(bpm, energy, has_bass, percussion_density,
                          spectral_centroid, rolloff)

    best_genre = None
    best_score = -1
    for genre, lo, hi, conds in _GENRE_PROFILES:
        if not (lo <= bpm <= hi):
            continue
        # Todas las condiciones deben cumplirse para ser candidato.
        ok = True
        for cond in conds:
            if not cond(f):
                ok = False
                break
        if not ok:
            continue
        score = len(conds)
        # `>` estricto: en empate gana el PRIMERO de la lista (prioridad).
        if score > best_score:
            best_score = score
            best_genre = genre

    return best_genre if best_genre is not None else _energy_fallback(f)


def get_genre_characteristics(genre: str) -> dict:
    """
    Retorna las características típicas de un género.
    Útil para sugerencias y validación.
    """
    characteristics = {
        'Techno': {'bpm_range': (120, 140), 'energy': 'high', 'bass': True},
        'House': {'bpm_range': (118, 130), 'energy': 'medium-high', 'bass': True},
        'Trance': {'bpm_range': (130, 150), 'energy': 'high', 'melodic': True},
        'Drum & Bass': {'bpm_range': (160, 180), 'energy': 'high', 'bass': True},
        'Dubstep': {'bpm_range': (140, 150), 'energy': 'high', 'bass': True},
        'Hip Hop': {'bpm_range': (70, 100), 'energy': 'medium', 'bass': True},
        'Trap': {'bpm_range': (130, 170), 'energy': 'medium-high', 'bass': True},
        'Reggaeton': {'bpm_range': (90, 100), 'energy': 'medium-high', 'bass': True},
        'Pop': {'bpm_range': (100, 130), 'energy': 'medium', 'melodic': True},
        'Rock': {'bpm_range': (100, 140), 'energy': 'high', 'distortion': True},
        'Metal': {'bpm_range': (100, 200), 'energy': 'very_high', 'distortion': True},
        'Jazz': {'bpm_range': (80, 200), 'energy': 'variable', 'melodic': True},
        'Ambient': {'bpm_range': (60, 100), 'energy': 'low', 'melodic': True},
    }

    return characteristics.get(genre, {'bpm_range': (60, 200), 'energy': 'variable'})
