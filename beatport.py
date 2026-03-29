"""
Beatport search and genre intelligence.
"""
import re
import json
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ==================== BEATPORT GENRE INTELLIGENCE ====================

BEATPORT_JUNK_GENRES = {
    'Mainstage', 'DJ Tools', 'Beats', 'Dance / Pop',
}

BEATPORT_QUALIFIER_TO_TYPE = {
    'Peak Time': 'peak_time', 'Driving': 'peak_time',
    'Raw': 'peak_time', 'Hard': 'peak_time',
    'Melodic': 'builder', 'Progressive': 'builder', 'Uplifting': 'builder',
    'Deep': 'opener', 'Hypnotic': 'opener', 'Minimal': 'opener', 'Deep Tech': 'opener',
    'Downtempo': 'warmup', 'Ambient': 'warmup', 'Organic': 'warmup',
    'Chill': 'warmup', 'Electronica': 'warmup',
    'Big Room': 'anthem', 'Electro House': 'anthem', 'Future Rave': 'anthem',
}


def clean_beatport_genre(raw_genre: str) -> dict:
    """
    Procesa un genero de Beatport y extrae genre limpio, track_type_hint, is_junk.
    """
    if not raw_genre:
        return {'genre': raw_genre, 'track_type_hint': None, 'is_junk': True}

    result = {
        'genre': raw_genre,
        'track_type_hint': None,
        'is_junk': raw_genre in BEATPORT_JUNK_GENRES,
    }

    paren_match = re.match(r'^(.+?)\s*\((.+)\)$', raw_genre)
    if paren_match:
        base_genre = paren_match.group(1).strip()
        qualifiers_str = paren_match.group(2).strip()
        result['genre'] = base_genre
        qualifiers = [q.strip() for q in qualifiers_str.replace('/', ',').split(',')]
        for q in qualifiers:
            q_clean = q.strip()
            if q_clean in BEATPORT_QUALIFIER_TO_TYPE:
                result['track_type_hint'] = BEATPORT_QUALIFIER_TO_TYPE[q_clean]
                break
    elif '/' in raw_genre and '(' not in raw_genre:
        result['genre'] = raw_genre
        parts = [p.strip() for p in raw_genre.split('/')]
        for p in parts:
            if p in BEATPORT_QUALIFIER_TO_TYPE:
                result['track_type_hint'] = BEATPORT_QUALIFIER_TO_TYPE[p]
                break

    return result


def convert_beatport_key(beatport_key: str) -> str:
    """Convierte key de formato Beatport ('D Minor') a estandar ('Dm')."""
    if not beatport_key:
        return None

    key = beatport_key.strip()
    key = key.replace(' Flat', 'b').replace(' flat', 'b')
    key = key.replace(' Sharp', '#').replace(' sharp', '#')
    key = key.replace(' Minor', 'm').replace(' minor', 'm')
    key = key.replace(' Major', '').replace(' major', '')
    key = key.replace(' min', 'm').replace(' maj', '')

    FLAT_TO_SHARP = {
        'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E',
        'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    }

    is_minor = key.endswith('m')
    base = key[:-1] if is_minor else key
    if base in FLAT_TO_SHARP:
        base = FLAT_TO_SHARP[base]

    return base + ('m' if is_minor else '')


def find_tracks_in_json(obj, results):
    """Busca recursivamente tracks en estructura JSON"""
    if isinstance(obj, dict):
        if 'bpm' in obj and ('name' in obj or 'title' in obj or 'track_name' in obj):
            results.append(obj)
        for v in obj.values():
            find_tracks_in_json(v, results)
    elif isinstance(obj, list):
        for item in obj:
            find_tracks_in_json(item, results)
    return results


def search_beatport(artist: str, title: str) -> Optional[Dict]:
    """
    Busca BPM y Key de un track en Beatport via scraping HTML.
    """
    try:
        import urllib.parse

        clean_title = re.sub(r'\s*\(?(Original Mix|Extended Mix|Radio Edit)\)?', '', title, flags=re.IGNORECASE).strip()
        clean_title = re.sub(r'^[A-D]\d\s+', '', clean_title).strip()
        clean_artist = artist.strip().rstrip('.')

        query = f"{clean_artist} {clean_title}"
        encoded_query = urllib.parse.quote(query)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }

        search_url = f"https://www.beatport.com/search?q={encoded_query}"

        response = requests.get(search_url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.warning(f"[Beatport] HTTP {response.status_code}")
            return None

        next_data_match = re.search(
            r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
            response.text, re.DOTALL
        )
        if not next_data_match:
            logger.warning(f"[Beatport] No __NEXT_DATA__ encontrado")
            return None

        data = json.loads(next_data_match.group(1))

        try:
            tracks = data["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]["tracks"]["data"]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"[Beatport] Estructura JSON no esperada")
            return None

        if not tracks:
            return None

        artist_lower = clean_artist.lower()
        title_lower = clean_title.lower()

        for track in tracks:
            if not isinstance(track, dict):
                continue

            track_name = track.get('track_name', '').lower()
            title_match = (title_lower in track_name) or (track_name in title_lower)
            if not title_match:
                title_words = set(title_lower.split())
                track_words = set(track_name.split())
                if title_words and track_words:
                    overlap = len(title_words & track_words) / max(len(title_words), 1)
                    title_match = overlap >= 0.6

            if not title_match:
                continue

            track_artists = track.get('artists', [])
            artist_match = False
            for a in track_artists:
                a_name = a.get('artist_name', '').lower() if isinstance(a, dict) else str(a).lower()
                if artist_lower in a_name or a_name in artist_lower:
                    artist_match = True
                    break

            if not artist_match and track_artists:
                continue

            result = {}

            if track.get('bpm'):
                try:
                    result['bpm'] = float(track['bpm'])
                except (ValueError, TypeError):
                    pass

            key_name = track.get('key_name', '')
            if key_name:
                result['key'] = convert_beatport_key(key_name)

            if track.get('length'):
                try:
                    result['duration'] = float(track['length']) / 1000.0
                except (ValueError, TypeError):
                    pass

            genres = track.get('genre', [])
            if genres and isinstance(genres, list):
                genre_names = [g.get('genre_name', '') for g in genres if isinstance(g, dict)]
                if genre_names:
                    raw_genre = genre_names[0]
                    cleaned = clean_beatport_genre(raw_genre)
                    result['genre'] = cleaned['genre']
                    result['genre_raw'] = raw_genre
                    result['is_junk_genre'] = cleaned['is_junk']
                    if cleaned['track_type_hint']:
                        result['track_type_hint'] = cleaned['track_type_hint']

            if result.get('bpm') or result.get('key'):
                return result

        return None

    except requests.exceptions.Timeout:
        logger.warning(f"[Beatport] Timeout")
        return None
    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        logger.error(f"[Beatport] Error: {e}")
        return None
