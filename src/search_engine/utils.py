# src/search_engine/utils.py

import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def pretty_persian(text):
    """Format Persian text for display"""
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)
