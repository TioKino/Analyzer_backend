"""
============================================================
DETECCIÓN DE GÉNEROS - DJ ANALYZER
============================================================

Sistema UNIVERSAL de detección y normalización de géneros.
Cubre todos los estilos musicales del mundo.
Integra Discogs, MusicBrainz y análisis espectral.

Total: ~400+ géneros mapeados
"""

import requests
import time
from typing import Optional

try:
    import discogs_client
    DISCOGS_AVAILABLE = True
except ImportError:
    DISCOGS_AVAILABLE = False
    print("⚠️ discogs_client no instalado. pip install discogs_client")


# ============================================================
# MAPEO UNIVERSAL DE GÉNEROS
# ============================================================

GENRE_MAP = {
    # ============================================================
    # TECHNO
    # ============================================================
    'techno': 'Techno',
    'tech': 'Techno',
    'detroit techno': 'Detroit Techno',
    'detroit': 'Detroit Techno',
    'minimal techno': 'Minimal Techno',
    'minimal': 'Minimal Techno',
    'acid techno': 'Acid Techno',
    'acid': 'Acid Techno',
    'industrial techno': 'Industrial Techno',
    'hard techno': 'Hard Techno',
    'peak time': 'Peak Time Techno',
    'peak time techno': 'Peak Time Techno',
    'melodic techno': 'Melodic Techno',
    'hypnotic': 'Hypnotic Techno',
    'hypnotic techno': 'Hypnotic Techno',
    'dub techno': 'Dub Techno',
    'dark techno': 'Dark Techno',
    'berlin techno': 'Berlin Techno',
    'raw techno': 'Raw Techno',
    'tribal techno': 'Tribal Techno',
    'atmospheric techno': 'Atmospheric Techno',
    'mental techno': 'Mental Techno',
    'bleep techno': 'Bleep Techno',
    'bleep': 'Bleep Techno',
    'ambient techno': 'Ambient Techno',
    'broken techno': 'Broken Techno',
    'schranz': 'Schranz',
    'birmingham techno': 'Birmingham Techno',
    'warehouse techno': 'Warehouse Techno',
    'rave techno': 'Rave Techno',
    
    # ============================================================
    # HOUSE
    # ============================================================
    'house': 'House',
    'deep house': 'Deep House',
    'deep': 'Deep House',
    'tech house': 'Tech House',
    'tech-house': 'Tech House',
    'progressive house': 'Progressive House',
    'prog house': 'Progressive House',
    'electro house': 'Electro House',
    'chicago house': 'Chicago House',
    'chicago': 'Chicago House',
    'afro house': 'Afro House',
    'afro': 'Afro House',
    'tribal house': 'Tribal House',
    'tribal': 'Tribal House',
    'funky house': 'Funky House',
    'funky': 'Funky House',
    'soulful house': 'Soulful House',
    'soulful': 'Soulful House',
    'minimal house': 'Minimal House',
    'jackin house': 'Jackin House',
    'jackin': 'Jackin House',
    'latin house': 'Latin House',
    'disco house': 'Disco House',
    'french house': 'French House',
    'filter house': 'Filter House',
    'future house': 'Future House',
    'melodic house': 'Melodic House',
    'bass house': 'Bass House',
    'organic house': 'Organic House',
    'italo house': 'Italo House',
    'balearic house': 'Balearic House',
    'balearic': 'Balearic House',
    'garage house': 'Garage House',
    'piano house': 'Piano House',
    'vocal house': 'Vocal House',
    'microhouse': 'Microhouse',
    'micro house': 'Microhouse',
    'lo-fi house': 'Lo-Fi House',
    'lofi house': 'Lo-Fi House',
    'outsider house': 'Outsider House',
    'ghetto house': 'Ghetto House',
    'hard house': 'Hard House',
    'nrg': 'NRG',
    'speed garage': 'Speed Garage',
    'mainroom house': 'Mainroom House',
    'mainroom': 'Mainroom House',
    'acid house': 'Acid House',
    
    # ============================================================
    # TRANCE
    # ============================================================
    'trance': 'Trance',
    'uplifting trance': 'Uplifting Trance',
    'uplifting': 'Uplifting Trance',
    'progressive trance': 'Progressive Trance',
    'prog trance': 'Progressive Trance',
    'psytrance': 'Psytrance',
    'psy trance': 'Psytrance',
    'psy': 'Psytrance',
    'goa': 'Goa Trance',
    'goa trance': 'Goa Trance',
    'vocal trance': 'Vocal Trance',
    'tech trance': 'Tech Trance',
    'hard trance': 'Hard Trance',
    'full on': 'Full On',
    'full-on': 'Full On',
    'darkpsy': 'Darkpsy',
    'dark psy': 'Darkpsy',
    'forest psy': 'Forest Psy',
    'forest': 'Forest Psy',
    'hi-tech': 'Hi-Tech',
    'hitech': 'Hi-Tech',
    'suomisaundi': 'Suomisaundi',
    'nitzhonot': 'Nitzhonot',
    'dream trance': 'Dream Trance',
    'euro trance': 'Euro Trance',
    'eurotrance': 'Euro Trance',
    'epic trance': 'Epic Trance',
    'balearic trance': 'Balearic Trance',
    'acid trance': 'Acid Trance',
    'classic trance': 'Classic Trance',
    'anjunabeats': 'Anjunabeats Style',
    'trouse': 'Trouse',
    'big room trance': 'Big Room Trance',
    
    # ============================================================
    # DRUM & BASS
    # ============================================================
    'drum & bass': 'Drum & Bass',
    'drum and bass': 'Drum & Bass',
    'drum&bass': 'Drum & Bass',
    'dnb': 'Drum & Bass',
    'd&b': 'Drum & Bass',
    "d'n'b": 'Drum & Bass',
    'liquid drum & bass': 'Liquid Drum & Bass',
    'liquid dnb': 'Liquid Drum & Bass',
    'liquid': 'Liquid Drum & Bass',
    'neurofunk': 'Neurofunk',
    'neuro': 'Neurofunk',
    'jump up': 'Jump Up',
    'jump-up': 'Jump Up',
    'jungle': 'Jungle',
    'darkstep': 'Darkstep',
    'techstep': 'Techstep',
    'ragga jungle': 'Ragga Jungle',
    'ragga': 'Ragga Jungle',
    'intelligent dnb': 'Intelligent DnB',
    'atmospheric dnb': 'Atmospheric DnB',
    'rollers': 'Rollers',
    'halftime': 'Halftime',
    'half-time': 'Halftime',
    'minimal dnb': 'Minimal DnB',
    'crossbreed': 'Crossbreed',
    'drumfunk': 'Drumfunk',
    'sambass': 'Sambass',
    'jazzstep': 'Jazzstep',
    'liquid funk': 'Liquid Funk',
    'dancefloor dnb': 'Dancefloor DnB',
    
    # ============================================================
    # DUBSTEP & BASS
    # ============================================================
    'dubstep': 'Dubstep',
    'brostep': 'Brostep',
    'deep dubstep': 'Deep Dubstep',
    'riddim': 'Riddim',
    'tearout': 'Tearout',
    'bass music': 'Bass Music',
    'bass': 'Bass Music',
    'future bass': 'Future Bass',
    'wave': 'Wave',
    'deathstep': 'Deathstep',
    'melodic dubstep': 'Melodic Dubstep',
    'chillstep': 'Chillstep',
    'post-dubstep': 'Post-Dubstep',
    'post dubstep': 'Post-Dubstep',
    'purple sound': 'Purple Sound',
    'hybrid trap': 'Hybrid Trap',
    'color bass': 'Color Bass',
    
    # ============================================================
    # UK SOUND
    # ============================================================
    'uk garage': 'UK Garage',
    'ukg': 'UK Garage',
    'garage': 'UK Garage',
    '2-step': '2-Step',
    '2 step': '2-Step',
    'two step': '2-Step',
    'bassline': 'Bassline',
    'uk funky': 'UK Funky',
    'grime': 'Grime',
    'uk bass': 'UK Bass',
    'night bass': 'Night Bass',
    'breaks': 'Breaks',
    'breakbeat': 'Breakbeat',
    'big beat': 'Big Beat',
    'nu skool breaks': 'Nu Skool Breaks',
    'nu skool': 'Nu Skool Breaks',
    'progressive breaks': 'Progressive Breaks',
    'florida breaks': 'Florida Breaks',
    'broken beat': 'Broken Beat',
    'bruk': 'Broken Beat',
    
    # ============================================================
    # HARD DANCE
    # ============================================================
    'hardstyle': 'Hardstyle',
    'hardcore': 'Hardcore',
    'gabber': 'Gabber',
    'gabba': 'Gabber',
    'happy hardcore': 'Happy Hardcore',
    'happy': 'Happy Hardcore',
    'frenchcore': 'Frenchcore',
    'terrorcore': 'Terrorcore',
    'speedcore': 'Speedcore',
    'industrial hardcore': 'Industrial Hardcore',
    'rawstyle': 'Rawstyle',
    'raw hardstyle': 'Rawstyle',
    'euphoric hardstyle': 'Euphoric Hardstyle',
    'euphoric': 'Euphoric Hardstyle',
    'hard nrg': 'Hard NRG',
    'freeform': 'Freeform',
    'freeform hardcore': 'Freeform',
    'uk hardcore': 'UK Hardcore',
    'makina': 'Makina',
    'hardtek': 'Hardtek',
    'tribe': 'Tribe',
    'uptempo hardcore': 'Uptempo Hardcore',
    'uptempo': 'Uptempo Hardcore',
    
    # ============================================================
    # ELECTRO
    # ============================================================
    'electro': 'Electro',
    'electro clash': 'Electro Clash',
    'electroclash': 'Electro Clash',
    'electropunk': 'Electropunk',
    'electro punk': 'Electropunk',
    'electro pop': 'Electro Pop',
    'detroit electro': 'Detroit Electro',
    'freestyle': 'Freestyle',
    'miami bass': 'Miami Bass',
    'ghetto tech': 'Ghetto Tech',
    'booty bass': 'Booty Bass',
    'electronica': 'Electronica',
    'art electronica': 'Art Electronica',
    
    # ============================================================
    # DISCO & FUNK
    # ============================================================
    'disco': 'Disco',
    'nu disco': 'Nu Disco',
    'nu-disco': 'Nu Disco',
    'italo disco': 'Italo Disco',
    'italo': 'Italo Disco',
    'funk': 'Funk',
    'boogie': 'Boogie',
    'space disco': 'Space Disco',
    'cosmic disco': 'Cosmic Disco',
    'cosmic': 'Cosmic Disco',
    'euro disco': 'Euro Disco',
    'eurodisco': 'Euro Disco',
    'hi-nrg': 'Hi-NRG',
    'high energy': 'Hi-NRG',
    'post-disco': 'Post-Disco',
    'disco polo': 'Disco Polo',
    'p-funk': 'P-Funk',
    'parliament': 'P-Funk',
    'g-funk': 'G-Funk',
    'electro funk': 'Electro Funk',
    'synth funk': 'Synth Funk',
    
    # ============================================================
    # AMBIENT & DOWNTEMPO
    # ============================================================
    'ambient': 'Ambient',
    'dark ambient': 'Dark Ambient',
    'drone': 'Drone',
    'downtempo': 'Downtempo',
    'chillout': 'Chillout',
    'chill out': 'Chillout',
    'chill': 'Chillout',
    'lounge': 'Lounge',
    'idm': 'IDM',
    'intelligent dance music': 'IDM',
    'glitch': 'Glitch',
    'experimental electronic': 'Experimental Electronic',
    'microsound': 'Microsound',
    'isolationism': 'Isolationism',
    'space music': 'Space Music',
    'new age electronic': 'New Age Electronic',
    'psybient': 'Psybient',
    'psychill': 'Psychill',
    'tribal ambient': 'Tribal Ambient',
    'field recordings': 'Field Recordings',
    
    # ============================================================
    # EDM & FESTIVAL
    # ============================================================
    'edm': 'EDM',
    'big room': 'Big Room',
    'bigroom': 'Big Room',
    'festival progressive': 'Festival Progressive',
    'bounce': 'Bounce',
    'melbourne bounce': 'Melbourne Bounce',
    'dutch house': 'Dutch House',
    'moombahton': 'Moombahton',
    'moombahcore': 'Moombahcore',
    'tropical house': 'Tropical House',
    'tropical': 'Tropical House',
    'slap house': 'Slap House',
    'brazilian bass': 'Brazilian Bass',
    'future bounce': 'Future Bounce',
    'dance pop': 'Dance Pop',
    
    # ============================================================
    # SYNTHWAVE & RETRO
    # ============================================================
    'synthwave': 'Synthwave',
    'synth wave': 'Synthwave',
    'retrowave': 'Retrowave',
    'retro wave': 'Retrowave',
    'darksynth': 'Darksynth',
    'dark synth': 'Darksynth',
    'outrun': 'Outrun',
    'dreamwave': 'Dreamwave',
    'chillwave': 'Chillwave',
    'vaporwave': 'Vaporwave',
    'future funk': 'Future Funk',
    'sovietwave': 'Sovietwave',
    'synthpop': 'Synthpop',
    'synth pop': 'Synthpop',
    'italo brutalo': 'Italo Brutalo',
    'cyberpunk': 'Cyberpunk',
    
    # ============================================================
    # INDUSTRIAL & EBM
    # ============================================================
    'industrial': 'Industrial',
    'ebm': 'EBM',
    'electronic body music': 'EBM',
    'dark electro': 'Dark Electro',
    'aggrotech': 'Aggrotech',
    'futurepop': 'Futurepop',
    'harsh ebm': 'Harsh EBM',
    'old school ebm': 'Old School EBM',
    'minimal wave': 'Minimal Wave',
    'coldwave': 'Coldwave',
    'cold wave': 'Coldwave',
    'witch house': 'Witch House',
    'darkwave': 'Darkwave',
    'dark wave': 'Darkwave',
    'ethereal wave': 'Ethereal Wave',
    'ethereal': 'Ethereal Wave',
    'post-industrial': 'Post-Industrial',
    
    # ============================================================
    # HIP HOP
    # ============================================================
    'hip hop': 'Hip Hop',
    'hip-hop': 'Hip Hop',
    'hiphop': 'Hip Hop',
    'rap': 'Rap',
    'boom bap': 'Boom Bap',
    'boombap': 'Boom Bap',
    'trap': 'Trap',
    'drill': 'Drill',
    'uk drill': 'UK Drill',
    'brooklyn drill': 'Brooklyn Drill',
    'chicago drill': 'Chicago Drill',
    'mumble rap': 'Mumble Rap',
    'cloud rap': 'Cloud Rap',
    'emo rap': 'Emo Rap',
    'alternative hip hop': 'Alternative Hip Hop',
    'alt hip hop': 'Alternative Hip Hop',
    'underground hip hop': 'Underground Hip Hop',
    'conscious hip hop': 'Conscious Hip Hop',
    'conscious rap': 'Conscious Hip Hop',
    'political hip hop': 'Political Hip Hop',
    'gangsta rap': 'Gangsta Rap',
    'gangster rap': 'Gangsta Rap',
    'west coast hip hop': 'West Coast Hip Hop',
    'west coast': 'West Coast Hip Hop',
    'east coast hip hop': 'East Coast Hip Hop',
    'east coast': 'East Coast Hip Hop',
    'southern hip hop': 'Southern Hip Hop',
    'dirty south': 'Dirty South',
    'crunk': 'Crunk',
    'snap music': 'Snap Music',
    'snap': 'Snap Music',
    'hyphy': 'Hyphy',
    'phonk': 'Phonk',
    'horrorcore': 'Horrorcore',
    'hardcore hip hop': 'Hardcore Hip Hop',
    'jazz rap': 'Jazz Rap',
    'lo-fi hip hop': 'Lo-Fi Hip Hop',
    'lofi hip hop': 'Lo-Fi Hip Hop',
    'lo fi': 'Lo-Fi Hip Hop',
    'abstract hip hop': 'Abstract Hip Hop',
    'trip hop': 'Trip Hop',
    'trip-hop': 'Trip Hop',
    'instrumental hip hop': 'Instrumental Hip Hop',
    'turntablism': 'Turntablism',
    
    # ============================================================
    # LATIN URBAN
    # ============================================================
    'reggaeton': 'Reggaeton',
    'regueton': 'Reggaeton',
    'trap latino': 'Trap Latino',
    'latin trap': 'Latin Trap',
    'dembow': 'Dembow',
    'perreo': 'Perreo',
    'spanish drill': 'Spanish Drill',
    'urbano latino': 'Urbano Latino',
    'urbano': 'Urbano Latino',
    'neo perreo': 'Neo Perreo',
    
    # ============================================================
    # ROCK
    # ============================================================
    'rock': 'Rock',
    'classic rock': 'Classic Rock',
    'hard rock': 'Hard Rock',
    'soft rock': 'Soft Rock',
    'blues rock': 'Blues Rock',
    'southern rock': 'Southern Rock',
    'heartland rock': 'Heartland Rock',
    'arena rock': 'Arena Rock',
    'glam rock': 'Glam Rock',
    'glam': 'Glam Rock',
    'psychedelic rock': 'Psychedelic Rock',
    'psych rock': 'Psychedelic Rock',
    'progressive rock': 'Progressive Rock',
    'prog rock': 'Progressive Rock',
    'prog': 'Progressive Rock',
    'art rock': 'Art Rock',
    'space rock': 'Space Rock',
    'krautrock': 'Krautrock',
    'stoner rock': 'Stoner Rock',
    'stoner': 'Stoner Rock',
    'desert rock': 'Desert Rock',
    'garage rock': 'Garage Rock',
    'surf rock': 'Surf Rock',
    'surf': 'Surf Rock',
    'rockabilly': 'Rockabilly',
    'rock and roll': 'Rock and Roll',
    'rock n roll': 'Rock and Roll',
    "rock'n'roll": 'Rock and Roll',
    'roots rock': 'Roots Rock',
    'americana rock': 'Americana Rock',
    'folk rock': 'Folk Rock',
    'country rock': 'Country Rock',
    'celtic rock': 'Celtic Rock',
    'latin rock': 'Latin Rock',
    'chicano rock': 'Chicano Rock',
    
    # ============================================================
    # ALTERNATIVE & INDIE
    # ============================================================
    'alternative rock': 'Alternative Rock',
    'alternative': 'Alternative Rock',
    'alt rock': 'Alternative Rock',
    'indie rock': 'Indie Rock',
    'indie': 'Indie Rock',
    'indie pop': 'Indie Pop',
    'britpop': 'Britpop',
    'brit pop': 'Britpop',
    'shoegaze': 'Shoegaze',
    'dream pop': 'Dream Pop',
    'noise pop': 'Noise Pop',
    'lo-fi': 'Lo-Fi',
    'slowcore': 'Slowcore',
    'sadcore': 'Sadcore',
    'post-rock': 'Post-Rock',
    'post rock': 'Post-Rock',
    'math rock': 'Math Rock',
    'noise rock': 'Noise Rock',
    'no wave': 'No Wave',
    'post-punk': 'Post-Punk',
    'post punk': 'Post-Punk',
    'gothic rock': 'Gothic Rock',
    'goth rock': 'Gothic Rock',
    'goth': 'Gothic Rock',
    'new wave': 'New Wave',
    'electropop': 'Electropop',
    'indietronica': 'Indietronica',
    'dance-punk': 'Dance-Punk',
    'dance punk': 'Dance-Punk',
    'post-punk revival': 'Post-Punk Revival',
    'garage rock revival': 'Garage Rock Revival',
    'twee pop': 'Twee Pop',
    'twee': 'Twee Pop',
    'chamber pop': 'Chamber Pop',
    'baroque pop': 'Baroque Pop',
    'jangle pop': 'Jangle Pop',
    'power pop': 'Power Pop',
    'college rock': 'College Rock',
    'madchester': 'Madchester',
    'baggy': 'Baggy',
    'c86': 'C86',
    
    # ============================================================
    # PUNK
    # ============================================================
    'punk rock': 'Punk Rock',
    'punk': 'Punk Rock',
    'hardcore punk': 'Hardcore Punk',
    'post-hardcore': 'Post-Hardcore',
    'post hardcore': 'Post-Hardcore',
    'melodic hardcore': 'Melodic Hardcore',
    'straight edge': 'Straight Edge',
    'sxe': 'Straight Edge',
    'crust punk': 'Crust Punk',
    'crust': 'Crust Punk',
    'd-beat': 'D-Beat',
    'grindcore': 'Grindcore',
    'grind': 'Grindcore',
    'powerviolence': 'Powerviolence',
    'screamo': 'Screamo',
    'emo': 'Emo',
    'pop punk': 'Pop Punk',
    'skate punk': 'Skate Punk',
    'street punk': 'Street Punk',
    'oi': 'Oi!',
    'oi!': 'Oi!',
    'anarcho punk': 'Anarcho Punk',
    'folk punk': 'Folk Punk',
    'cowpunk': 'Cowpunk',
    'psychobilly': 'Psychobilly',
    'horror punk': 'Horror Punk',
    'deathrock': 'Deathrock',
    'ska punk': 'Ska Punk',
    'ska core': 'Ska Core',
    'skacore': 'Ska Core',
    
    # ============================================================
    # METAL
    # ============================================================
    'heavy metal': 'Heavy Metal',
    'metal': 'Heavy Metal',
    'thrash metal': 'Thrash Metal',
    'thrash': 'Thrash Metal',
    'death metal': 'Death Metal',
    'black metal': 'Black Metal',
    'doom metal': 'Doom Metal',
    'doom': 'Doom Metal',
    'power metal': 'Power Metal',
    'speed metal': 'Speed Metal',
    'progressive metal': 'Progressive Metal',
    'prog metal': 'Progressive Metal',
    'symphonic metal': 'Symphonic Metal',
    'gothic metal': 'Gothic Metal',
    'folk metal': 'Folk Metal',
    'viking metal': 'Viking Metal',
    'pagan metal': 'Pagan Metal',
    'industrial metal': 'Industrial Metal',
    'nu metal': 'Nu Metal',
    'nu-metal': 'Nu Metal',
    'alternative metal': 'Alternative Metal',
    'alt metal': 'Alternative Metal',
    'groove metal': 'Groove Metal',
    'metalcore': 'Metalcore',
    'deathcore': 'Deathcore',
    'djent': 'Djent',
    'post-metal': 'Post-Metal',
    'post metal': 'Post-Metal',
    'sludge metal': 'Sludge Metal',
    'sludge': 'Sludge Metal',
    'stoner metal': 'Stoner Metal',
    'drone metal': 'Drone Metal',
    'avant-garde metal': 'Avant-Garde Metal',
    'technical death metal': 'Technical Death Metal',
    'tech death': 'Technical Death Metal',
    'melodic death metal': 'Melodic Death Metal',
    'melodeath': 'Melodic Death Metal',
    'brutal death metal': 'Brutal Death Metal',
    'slam death metal': 'Slam Death Metal',
    'slam': 'Slam Death Metal',
    'atmospheric black metal': 'Atmospheric Black Metal',
    'depressive black metal': 'Depressive Black Metal',
    'dsbm': 'Depressive Black Metal',
    'symphonic black metal': 'Symphonic Black Metal',
    'blackgaze': 'Blackgaze',
    'funeral doom': 'Funeral Doom',
    'epic doom': 'Epic Doom',
    'traditional doom': 'Traditional Doom',
    'nwobhm': 'NWOBHM',
    'glam metal': 'Glam Metal',
    'hair metal': 'Hair Metal',
    'crossover thrash': 'Crossover Thrash',
    'crossover': 'Crossover Thrash',
    'goregrind': 'Goregrind',
    'pornogrind': 'Pornogrind',
    'mathcore': 'Mathcore',
    'noisegrind': 'Noisegrind',
    
    # ============================================================
    # POP
    # ============================================================
    'pop': 'Pop',
    'synth pop': 'Synth Pop',
    'teen pop': 'Teen Pop',
    'bubblegum pop': 'Bubblegum Pop',
    'bubblegum': 'Bubblegum Pop',
    'art pop': 'Art Pop',
    'experimental pop': 'Experimental Pop',
    'hyperpop': 'Hyperpop',
    'hyper pop': 'Hyperpop',
    'sophisti-pop': 'Sophisti-Pop',
    'sunshine pop': 'Sunshine Pop',
    'yacht rock': 'Yacht Rock',
    'adult contemporary': 'Adult Contemporary',
    'easy listening': 'Easy Listening',
    'soft pop': 'Soft Pop',
    'pop rock': 'Pop Rock',
    'emo pop': 'Emo Pop',
    'k-pop': 'K-Pop',
    'kpop': 'K-Pop',
    'j-pop': 'J-Pop',
    'jpop': 'J-Pop',
    'c-pop': 'C-Pop',
    'cpop': 'C-Pop',
    'cantopop': 'Cantopop',
    'mandopop': 'Mandopop',
    'latin pop': 'Latin Pop',
    'euro pop': 'Euro Pop',
    'europop': 'Europop',
    'italo pop': 'Italo Pop',
    'city pop': 'City Pop',
    'future pop': 'Future Pop',
    
    # ============================================================
    # R&B & SOUL
    # ============================================================
    'r&b': 'R&B',
    'rnb': 'R&B',
    'rhythm and blues': 'R&B',
    'soul': 'Soul',
    'neo soul': 'Neo Soul',
    'neo-soul': 'Neo Soul',
    'contemporary r&b': 'Contemporary R&B',
    'alternative r&b': 'Alternative R&B',
    'pbr&b': 'PBR&B',
    'new jack swing': 'New Jack Swing',
    'quiet storm': 'Quiet Storm',
    'motown': 'Motown',
    'northern soul': 'Northern Soul',
    'southern soul': 'Southern Soul',
    'deep soul': 'Deep Soul',
    'blue-eyed soul': 'Blue-Eyed Soul',
    'philly soul': 'Philly Soul',
    'philadelphia soul': 'Philly Soul',
    'chicago soul': 'Chicago Soul',
    'memphis soul': 'Memphis Soul',
    'psychedelic soul': 'Psychedelic Soul',
    'go-go': 'Go-Go',
    'doo-wop': 'Doo-Wop',
    'doowop': 'Doo-Wop',
    'gospel': 'Gospel',
    'contemporary gospel': 'Contemporary Gospel',
    'urban contemporary': 'Urban Contemporary',
    
    # ============================================================
    # JAZZ
    # ============================================================
    'jazz': 'Jazz',
    'bebop': 'Bebop',
    'be-bop': 'Bebop',
    'hard bop': 'Hard Bop',
    'cool jazz': 'Cool Jazz',
    'modal jazz': 'Modal Jazz',
    'free jazz': 'Free Jazz',
    'avant-garde jazz': 'Avant-Garde Jazz',
    'fusion': 'Fusion',
    'jazz fusion': 'Jazz Fusion',
    'jazz funk': 'Jazz Funk',
    'soul jazz': 'Soul Jazz',
    'acid jazz': 'Acid Jazz',
    'smooth jazz': 'Smooth Jazz',
    'contemporary jazz': 'Contemporary Jazz',
    'nu jazz': 'Nu Jazz',
    'jazz house': 'Jazz House',
    'electro jazz': 'Electro Jazz',
    'swing': 'Swing',
    'big band': 'Big Band',
    'dixieland': 'Dixieland',
    'new orleans jazz': 'New Orleans Jazz',
    'chicago jazz': 'Chicago Jazz',
    'west coast jazz': 'West Coast Jazz',
    'third stream': 'Third Stream',
    'chamber jazz': 'Chamber Jazz',
    'vocal jazz': 'Vocal Jazz',
    'latin jazz': 'Latin Jazz',
    'afro-cuban jazz': 'Afro-Cuban Jazz',
    'bossa nova jazz': 'Bossa Nova Jazz',
    'gypsy jazz': 'Gypsy Jazz',
    'ethio-jazz': 'Ethio-Jazz',
    'spiritual jazz': 'Spiritual Jazz',
    'post-bop': 'Post-Bop',
    'm-base': 'M-Base',
    
    # ============================================================
    # BLUES
    # ============================================================
    'blues': 'Blues',
    'delta blues': 'Delta Blues',
    'chicago blues': 'Chicago Blues',
    'electric blues': 'Electric Blues',
    'texas blues': 'Texas Blues',
    'west coast blues': 'West Coast Blues',
    'memphis blues': 'Memphis Blues',
    'piedmont blues': 'Piedmont Blues',
    'country blues': 'Country Blues',
    'acoustic blues': 'Acoustic Blues',
    'british blues': 'British Blues',
    'jump blues': 'Jump Blues',
    'boogie woogie': 'Boogie Woogie',
    'soul blues': 'Soul Blues',
    'modern blues': 'Modern Blues',
    
    # ============================================================
    # COUNTRY & FOLK
    # ============================================================
    'country': 'Country',
    'classic country': 'Classic Country',
    'outlaw country': 'Outlaw Country',
    'alt-country': 'Alt-Country',
    'alt country': 'Alt-Country',
    'americana': 'Americana',
    'country pop': 'Country Pop',
    'bro-country': 'Bro-Country',
    'country rap': 'Country Rap',
    'bluegrass': 'Bluegrass',
    'progressive bluegrass': 'Progressive Bluegrass',
    'newgrass': 'Newgrass',
    'old-time': 'Old-Time',
    'honky tonk': 'Honky Tonk',
    'western swing': 'Western Swing',
    'bakersfield sound': 'Bakersfield Sound',
    'nashville sound': 'Nashville Sound',
    'texas country': 'Texas Country',
    'red dirt': 'Red Dirt',
    'folk': 'Folk',
    'contemporary folk': 'Contemporary Folk',
    'anti-folk': 'Anti-Folk',
    'freak folk': 'Freak Folk',
    'psych folk': 'Psych Folk',
    'neofolk': 'Neofolk',
    'neo-folk': 'Neofolk',
    'industrial folk': 'Industrial Folk',
    'celtic folk': 'Celtic Folk',
    'british folk': 'British Folk',
    'american folk': 'American Folk',
    'appalachian folk': 'Appalachian Folk',
    'protest folk': 'Protest Folk',
    'singer-songwriter': 'Singer-Songwriter',
    
    # ============================================================
    # REGGAE & DUB
    # ============================================================
    'reggae': 'Reggae',
    'roots reggae': 'Roots Reggae',
    'roots': 'Roots Reggae',
    'dub': 'Dub',
    'lovers rock': 'Lovers Rock',
    'dancehall': 'Dancehall',
    'digital reggae': 'Digital Reggae',
    'rocksteady': 'Rocksteady',
    'ska': 'Ska',
    'two-tone': 'Two-Tone',
    '2 tone': 'Two-Tone',
    'reggae fusion': 'Reggae Fusion',
    'digital dub': 'Digital Dub',
    'steppers': 'Steppers',
    
    # ============================================================
    # CARIBBEAN
    # ============================================================
    'soca': 'Soca',
    'calypso': 'Calypso',
    'zouk': 'Zouk',
    'compas': 'Compas',
    'kompa': 'Compas',
    'merengue': 'Merengue',
    'bachata': 'Bachata',
    'salsa': 'Salsa',
    'cumbia': 'Cumbia',
    'vallenato': 'Vallenato',
    'punta': 'Punta',
    
    # ============================================================
    # LATIN
    # ============================================================
    'latin': 'Latin',
    'cumbia villera': 'Cumbia Villera',
    'cumbia digital': 'Cumbia Digital',
    'bossa nova': 'Bossa Nova',
    'mpb': 'MPB',
    'samba': 'Samba',
    'pagode': 'Pagode',
    'axe': 'Axé',
    'forro': 'Forró',
    'forró': 'Forró',
    'sertanejo': 'Sertanejo',
    'funk carioca': 'Funk Carioca',
    'baile funk': 'Baile Funk',
    'tecnobrega': 'Tecnobrega',
    'tango': 'Tango',
    'tango electronico': 'Tango Electrónico',
    'nuevo tango': 'Nuevo Tango',
    'ranchera': 'Ranchera',
    'mariachi': 'Mariachi',
    'norteno': 'Norteño',
    'norteño': 'Norteño',
    'banda': 'Banda',
    'corridos': 'Corridos',
    'regional mexicano': 'Regional Mexicano',
    'tejano': 'Tejano',
    'musica tropical': 'Música Tropical',
    'son cubano': 'Son Cubano',
    'timba': 'Timba',
    'bolero': 'Bolero',
    'trova': 'Trova',
    'nueva cancion': 'Nueva Canción',
    'andean': 'Andean',
    'huayno': 'Huayno',
    'chicha': 'Chicha',
    
    # ============================================================
    # AFRICAN
    # ============================================================
    'afrobeats': 'Afrobeats',
    'afropop': 'Afropop',
    'amapiano': 'Amapiano',
    'gqom': 'Gqom',
    'kwaito': 'Kwaito',
    'highlife': 'Highlife',
    'hiplife': 'Hiplife',
    'juju': 'Jùjú',
    'fuji': 'Fuji',
    'afrobeat': 'Afrobeat',
    'mbalax': 'Mbalax',
    'soukous': 'Soukous',
    'rumba congolaise': 'Rumba Congolaise',
    'ndombolo': 'Ndombolo',
    'coupe-decale': 'Coupé-Décalé',
    'kuduro': 'Kuduro',
    'kizomba': 'Kizomba',
    'semba': 'Semba',
    'bikutsi': 'Bikutsi',
    'makossa': 'Makossa',
    'benga': 'Benga',
    'bongo flava': 'Bongo Flava',
    'taarab': 'Taarab',
    'gnawa': 'Gnawa',
    'rai': 'Rai',
    'chaabi': 'Chaabi',
    'shangaan electro': 'Shangaan Electro',
    'south african house': 'South African House',
    'afro tech': 'Afro Tech',
    
    # ============================================================
    # ASIAN
    # ============================================================
    'k-hip hop': 'K-Hip Hop',
    'k-r&b': 'K-R&B',
    'k-rock': 'K-Rock',
    'k-indie': 'K-Indie',
    'trot': 'Trot',
    'j-rock': 'J-Rock',
    'visual kei': 'Visual Kei',
    'shibuya-kei': 'Shibuya-kei',
    'enka': 'Enka',
    'c-rock': 'C-Rock',
    'chinese hip hop': 'Chinese Hip Hop',
    'bollywood': 'Bollywood',
    'indian pop': 'Indian Pop',
    'indian classical': 'Indian Classical',
    'carnatic': 'Carnatic',
    'hindustani': 'Hindustani',
    'filmi': 'Filmi',
    'bhangra': 'Bhangra',
    'indian electronic': 'Indian Electronic',
    'thai pop': 'Thai Pop',
    't-pop': 'T-Pop',
    'v-pop': 'V-Pop',
    'vietnamese pop': 'Vietnamese Pop',
    'indo pop': 'Indo Pop',
    'dangdut': 'Dangdut',
    'opm': 'OPM',
    'pinoy pop': 'Pinoy Pop',
    
    # ============================================================
    # WORLD
    # ============================================================
    'world music': 'World Music',
    'world fusion': 'World Fusion',
    'global bass': 'Global Bass',
    'worldbeat': 'Worldbeat',
    'ethnic electronica': 'Ethnic Electronica',
    'oriental': 'Oriental',
    'middle eastern': 'Middle Eastern',
    'arabic pop': 'Arabic Pop',
    'turkish pop': 'Turkish Pop',
    'arabesque': 'Arabesque',
    'persian pop': 'Persian Pop',
    'israeli pop': 'Israeli Pop',
    'klezmer': 'Klezmer',
    'balkan': 'Balkan',
    'balkan beat': 'Balkan Beat',
    'turbo folk': 'Turbo Folk',
    'greek': 'Greek',
    'rebetiko': 'Rebetiko',
    'laiko': 'Laiko',
    'flamenco': 'Flamenco',
    'nuevo flamenco': 'Nuevo Flamenco',
    'fado': 'Fado',
    'celtic': 'Celtic',
    'irish folk': 'Irish Folk',
    'scottish': 'Scottish',
    'nordic': 'Nordic',
    'scandinavian': 'Scandinavian',
    'russian': 'Russian',
    'slavic': 'Slavic',
    'gypsy': 'Gypsy',
    'romani': 'Romani',
    'polka': 'Polka',
    'australian': 'Australian',
    'didgeridoo': 'Didgeridoo',
    'polynesian': 'Polynesian',
    'hawaiian': 'Hawaiian',
    
    # ============================================================
    # CLASSICAL
    # ============================================================
    'classical': 'Classical',
    'baroque': 'Baroque',
    'classical period': 'Classical Period',
    'romantic': 'Romantic',
    'impressionist': 'Impressionist',
    'modern classical': 'Modern Classical',
    'contemporary classical': 'Contemporary Classical',
    'minimalist': 'Minimalist',
    'post-minimalist': 'Post-Minimalist',
    'neoclassical': 'Neoclassical',
    'neo-romantic': 'Neo-Romantic',
    'avant-garde classical': 'Avant-Garde Classical',
    'spectral': 'Spectral',
    'electroacoustic': 'Electroacoustic',
    'acousmatic': 'Acousmatic',
    'musique concrete': 'Musique Concrète',
    'opera': 'Opera',
    'operetta': 'Operetta',
    'choral': 'Choral',
    'sacred music': 'Sacred Music',
    'chamber music': 'Chamber Music',
    'symphony': 'Symphony',
    'orchestral': 'Orchestral',
    'concerto': 'Concerto',
    'sonata': 'Sonata',
    'string quartet': 'String Quartet',
    'piano solo': 'Piano Solo',
    'early music': 'Early Music',
    'medieval': 'Medieval',
    'renaissance': 'Renaissance',
    'gregorian chant': 'Gregorian Chant',
    
    # ============================================================
    # SOUNDTRACK
    # ============================================================
    'soundtrack': 'Soundtrack',
    'ost': 'Soundtrack',
    'film score': 'Film Score',
    'tv score': 'TV Score',
    'video game music': 'Video Game Music',
    'vgm': 'Video Game Music',
    'anime soundtrack': 'Anime Soundtrack',
    'musical theatre': 'Musical Theatre',
    'broadway': 'Broadway',
    'west end': 'West End',
    'jingle': 'Jingle',
    'library music': 'Library Music',
    'production music': 'Production Music',
    'trailer music': 'Trailer Music',
    'epic music': 'Epic Music',
    'cinematic': 'Cinematic',
    'orchestral soundtrack': 'Orchestral Soundtrack',
    'electronic score': 'Electronic Score',
    'synthwave score': 'Synthwave Score',
    
    # ============================================================
    # EXPERIMENTAL
    # ============================================================
    'experimental': 'Experimental',
    'avant-garde': 'Avant-Garde',
    'noise': 'Noise',
    'power electronics': 'Power Electronics',
    'harsh noise': 'Harsh Noise',
    'harsh noise wall': 'Harsh Noise Wall',
    'hnw': 'Harsh Noise Wall',
    'japanoise': 'Japanoise',
    'sound collage': 'Sound Collage',
    'plunderphonics': 'Plunderphonics',
    'tape music': 'Tape Music',
    'free improvisation': 'Free Improvisation',
    'improv': 'Improv',
    'sound art': 'Sound Art',
    'installation': 'Installation',
    'lowercase': 'Lowercase',
    'onkyo': 'Onkyo',
    'reductionism': 'Reductionism',
    'eai': 'EAI',
    'clicks & cuts': 'Clicks & Cuts',
    'generative': 'Generative',
    'algorithmic': 'Algorithmic',
    'aleatoric': 'Aleatoric',
    'spectral music': 'Spectral Music',
    
    # ============================================================
    # SPOKEN & OTHER
    # ============================================================
    'spoken word': 'Spoken Word',
    'poetry': 'Poetry',
    'slam poetry': 'Slam Poetry',
    'audiobook': 'Audiobook',
    'radio drama': 'Radio Drama',
    'comedy': 'Comedy',
    'stand-up': 'Stand-Up',
    'podcast': 'Podcast',
    'asmr': 'ASMR',
    'field recording': 'Field Recording',
    'nature sounds': 'Nature Sounds',
    'meditation': 'Meditation',
    'binaural': 'Binaural',
    'healing': 'Healing',
    'new age': 'New Age',
    'worship': 'Worship',
    'christian': 'Christian',
    'praise': 'Praise',
    'hymns': 'Hymns',
    'religious': 'Religious',
    'spiritual': 'Spiritual',
    'devotional': 'Devotional',
    'children music': 'Children Music',
    'nursery rhymes': 'Nursery Rhymes',
    'educational': 'Educational',
    'holiday': 'Holiday',
    'christmas': 'Christmas',
    'halloween': 'Halloween',
    'novelty': 'Novelty',
    'parody': 'Parody',
}


class GenreDetector:
    def __init__(self, discogs_token=None):
        self.discogs_token = discogs_token
        self.discogs_client = None
        
        if discogs_token and DISCOGS_AVAILABLE:
            try:
                self.discogs_client = discogs_client.Client(
                    'DJAnalyzerPro/2.3',
                    user_token=discogs_token
                )
            except Exception as e:
                print(f"⚠️ Error inicializando Discogs: {e}")
    
    def get_discogs_genre(self, artist: str, title: str) -> Optional[dict]:
        """Obtener género de Discogs"""
        if not self.discogs_client or not artist or not title:
            return None
        
        try:
            # Buscar release
            query = f"{artist} {title}"
            results = self.discogs_client.search(query, type='release')
            
            # Iterar sobre resultados (evita el error de slicing)
            release = None
            for r in results:
                release = r
                break
            
            if not release:
                return None
            
            # Obtener géneros
            genres = getattr(release, 'genres', None)
            styles = getattr(release, 'styles', None)
            
            # Obtener label y year si están disponibles
            label = None
            year = None
            try:
                if hasattr(release, 'labels') and release.labels:
                    label = release.labels[0].name if hasattr(release.labels[0], 'name') else str(release.labels[0])
                if hasattr(release, 'year'):
                    year = release.year
            except:
                pass
            
            if styles:
                # Usar estilo (más específico)
                return {
                    'genre': self._normalize_genre(styles[0]),
                    'confidence': 0.85,
                    'source': 'discogs',
                    'parent_genre': self._normalize_genre(genres[0]) if genres else None,
                    'label': label,
                    'year': year
                }
            elif genres:
                return {
                    'genre': self._normalize_genre(genres[0]),
                    'confidence': 0.7,
                    'source': 'discogs',
                    'label': label,
                    'year': year
                }
            
            return None
            
        except Exception as e:
            print(f"Error Discogs: {e}")
            return None
    
    def get_musicbrainz_info(self, artist: str, title: str) -> Optional[dict]:
        """Obtener info de MusicBrainz"""
        if not artist or not title:
            return None
        
        try:
            # Buscar en MusicBrainz
            search_url = "https://musicbrainz.org/ws/2/recording/"
            params = {
                'query': f'artist:"{artist}" AND recording:"{title}"',
                'fmt': 'json',
                'limit': 1
            }
            
            headers = {'User-Agent': 'DJAnalyzerPro/2.3 (contact@djanalyzer.pro)'}
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            recordings = data.get('recordings', [])
            
            if not recordings:
                return None
            
            recording = recordings[0]
            
            # Extraer tags como género
            tags = recording.get('tags', [])
            genre = None
            if tags:
                # Ordenar por count y tomar el primero
                sorted_tags = sorted(tags, key=lambda x: x.get('count', 0), reverse=True)
                genre = self._normalize_genre(sorted_tags[0].get('name', ''))
            
            # Rate limit para MusicBrainz (1 req/sec)
            time.sleep(1)
            
            return {
                'genre': genre,
                'mbid': recording.get('id'),
                'source': 'musicbrainz'
            }
            
        except Exception as e:
            print(f"Error MusicBrainz: {e}")
            return None
    
    def _normalize_genre(self, genre: str) -> str:
        """Normalizar géneros a categorías consistentes usando el mapeo universal"""
        if not genre:
            return 'Electronic'
            
        genre_lower = genre.lower().strip()
        
        # Buscar coincidencia exacta primero
        if genre_lower in GENRE_MAP:
            return GENRE_MAP[genre_lower]
        
        # Buscar coincidencias parciales
        for key, value in GENRE_MAP.items():
            if key in genre_lower:
                return value
        
        # Si no hay match, capitalizar
        return genre.title()
    
    def detect_genre(self, artist: str, title: str, 
                    collective_genre: Optional[str] = None,
                    spectral_genre: str = "Electronic") -> dict:
        """
        Detectar género con sistema de prioridades
        
        Prioridad:
        1. Memoria colectiva (correcciones de usuarios)
        2. Discogs (más específico para electrónica)
        3. MusicBrainz
        4. Análisis espectral
        """
        
        # 1. Memoria colectiva (máxima prioridad)
        if collective_genre:
            return {
                'genre': collective_genre,
                'confidence': 1.0,
                'source': 'collective_memory'
            }
        
        # 2. Discogs (mejor para electrónica)
        discogs_result = self.get_discogs_genre(artist, title)
        if discogs_result:
            return discogs_result
        
        # 3. MusicBrainz
        mb_result = self.get_musicbrainz_info(artist, title)
        if mb_result and mb_result.get('genre'):
            return mb_result
        
        # 4. Fallback: Análisis espectral
        return {
            'genre': spectral_genre,
            'confidence': 0.5,
            'source': 'spectral_analysis'
        }


def get_all_genres() -> list:
    """Retorna lista de todos los géneros normalizados únicos"""
    return sorted(list(set(GENRE_MAP.values())))


def get_genre_count() -> int:
    """Retorna el número total de géneros únicos"""
    return len(set(GENRE_MAP.values()))


def search_genres(query: str) -> list:
    """Busca géneros que contengan el texto dado"""
    if not query:
        return []
    query_lower = query.lower()
    all_genres = set(GENRE_MAP.values())
    return sorted([g for g in all_genres if query_lower in g.lower()])
