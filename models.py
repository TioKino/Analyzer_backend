"""
Pydantic models and key mappings for DJ Analyzer Pro API.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel


class AnalysisResult(BaseModel):
    model_config = {"extra": "ignore"}

    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    label: Optional[str] = None
    year: Optional[str] = None
    isrc: Optional[str] = None
    duration: float
    bpm: float
    bpm_confidence: float
    bpm_source: str = "analysis"
    key: Optional[str] = None
    camelot: Optional[str] = None
    key_confidence: float
    key_source: str = "analysis"
    energy_raw: float
    energy_normalized: float
    energy_dj: int
    groove_score: float
    swing_factor: float
    has_intro: bool
    has_buildup: bool
    has_drop: bool
    has_breakdown: bool
    has_outro: bool
    structure_sections: List[Dict]
    track_type: str
    track_type_source: str = "waveform"
    genre: str = "unknown"
    subgenre: Optional[str] = None
    genre_source: str = "spectral_analysis"
    has_vocals: bool
    has_heavy_bass: bool
    has_pads: bool
    percussion_density: float
    mix_energy_start: float
    mix_energy_end: float
    drop_timestamp: float
    cue_points: List[Dict] = []
    first_beat: float = 0.0
    beat_interval: float = 0.5
    artwork_embedded: bool = False
    artwork_url: Optional[str] = None
    preview_url: Optional[str] = None
    fingerprint: Optional[str] = None


class CorrectionRequest(BaseModel):
    track_id: str
    field: str
    old_value: str
    new_value: str
    fingerprint: Optional[str] = None


KEY_TO_CAMELOT = {
    'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B',
    'E': '12B', 'F': '7B', 'F#': '2B', 'G': '9B',
    'G#': '4B', 'A': '11B', 'A#': '6B', 'B': '1B',
    'Cm': '5A', 'C#m': '12A', 'Dm': '7A', 'D#m': '2A',
    'Em': '9A', 'Fm': '4A', 'F#m': '11A', 'Gm': '6A',
    'G#m': '1A', 'Am': '8A', 'A#m': '3A', 'Bm': '10A'
}


def get_camelot(key: str) -> str:
    """Convierte key musical a notacion Camelot"""
    return KEY_TO_CAMELOT.get(key, '?')
