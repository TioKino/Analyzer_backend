"""
DJ ANALYZER - Artwork & Cue Points Module v2.3.0
====================================================
Modulo para extraccion de artwork y deteccion de cue points.

Exporta:
- extract_artwork_from_file()
- extract_id3_metadata()
- detect_cue_points()
- detect_beat_grid()
- save_artwork_to_cache()
- search_artwork_online()
- ARTWORK_CACHE_DIR
"""
from config import LASTFM_API_KEY, ARTWORK_CACHE_DIR
import logging
import os
import base64
import requests
from typing import Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)
os.makedirs(ARTWORK_CACHE_DIR, exist_ok=True)
