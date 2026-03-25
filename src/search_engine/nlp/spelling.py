# src/search_engine/nlp/spelling.py

import logging
from typing import List, Dict, Tuple
from parsivar import SpellCheck
from src.search_engine.config import INDEX_NAME

# Configure logger
logger = logging.getLogger(__name__)

# Standard Iranian keyboard mapping
ENGLISH_TO_PERSIAN_MAP = {
    '`': '÷', '1': '۱', '2': '۲', '3': '۳', '4': '۴', '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹', '0': '۰', '-': '-', '=': '=',
    'q': 'ض', 'w': 'ص', 'e': 'ث', 'r': 'ق', 't': 'ف', 'y': 'غ', 'u': 'ع', 'i': 'ه', 'o': 'خ', 'p': 'ح', '[': 'ج', ']': 'چ',
    'a': 'ش', 's': 'س', 'd': 'ی', 'f': 'ب', 'g': 'ل', 'h': 'ا', 'j': 'ت', 'k': 'ن', 'l': 'م', ';': 'ک', "'": 'گ',
    'z': 'ظ', 'x': 'ط', 'c': 'ز', 'v': 'ر', 'b': 'ذ', 'n': 'د', 'm': 'پ', ',': 'و', '.': '.', '/': '/',
}

def is_likely_mistyped_persian(text: str) -> bool:
    if not text: return False
    clean_text = ''.join(c for c in text if c.isalpha())
    if not clean_text: return False
    has_english = any('a' <= c.lower() <= 'z' for c in clean_text)
    if not has_english: return False
    persian_patterns = ['hk', ';', 'vs', 'jhf', 'sj', 'dh', 'fn', "'"]
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in persian_patterns): return True
    persian_common_keys = ';hjklfgnsd'
    common_key_count = sum(1 for c in text_lower if c in persian_common_keys)
    if len(clean_text) > 0 and common_key_count / len(clean_text) > 0.4: return True
    return False

def convert_english_to_persian(text: str) -> str:
    return ''.join(ENGLISH_TO_PERSIAN_MAP.get(char, char) for char in text)

def detect_and_convert_keyboard_layout(text: str) -> tuple:
    if not text or not text.strip(): return text, False, 'empty'
    has_persian = any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF' for c in text)
    if has_persian: return text, False, 'already_persian'
    if is_likely_mistyped_persian(text):
        converted = convert_english_to_persian(text)
        logger.info(f"Keyboard conversion: '{text}' -> '{converted}'")
        return converted, True, 'mistyped_persian'
    return text, False, 'no_conversion'

class Speller:
    def __init__(self, es_client=None):
        self.es_client = es_client
        self.spell_checker = None # Lazy init

    def initialize(self):
        if not self.spell_checker:
            logger.info("Initializing Parsivar SpellCheck...")
            self.spell_checker = SpellCheck()

    def suggest_corrections(self, query_text: str, max_suggestions: int = 3) -> List[Dict]:
        self.initialize()
        suggestions = []
        try:
            logger.debug(f"Checking spelling for: {query_text}")
            corrected = self.spell_checker.spell_corrector(query_text)
            if corrected != query_text:
                logger.info(f"Spell correction: '{query_text}' -> '{corrected}'")
                suggestions.append({'text': corrected, 'confidence': 0.8, 'source': 'parsivar'})
        except Exception as e:
            logger.warning(f"Spell correction failed: {e}")
        return suggestions[:max_suggestions]
