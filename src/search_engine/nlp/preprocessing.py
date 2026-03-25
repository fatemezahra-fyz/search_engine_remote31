# src/search_engine/nlp/preprocessing.py

import re
import logging
from typing import List
from parsivar import Normalizer, Tokenizer, FindStems

# Configure logger
logger = logging.getLogger(__name__)

# Move NLP objects to module-level singletons
logger.info("Initializing Parsivar NLP components...")
normalizer = Normalizer()
tokenizer = Tokenizer()
stemmer = FindStems()

def clean_query_for_llm(query: str) -> str:
    """
    Minimal query cleaning optimized for LLM consumption.
    Preserves descriptive adjectives that the LLM uses for technical mapping.
    """
    logger.debug(f"Cleaning query for LLM: {query}")
    normalized = normalizer.normalize(query)
    return normalized

def clean_query_for_retrieval(query: str) -> str:
    """
    Comprehensive query cleaning for traditional retrieval (BM25/kNN).
    Includes normalization, stop word removal, tokenization, and stemming.
    """
    logger.debug(f"Cleaning query for retrieval: {query}")
    normalized = normalizer.normalize(query)

    # Persian stop words for e-commerce
    persian_stopwords = {
        'و', 'یا', 'در', 'به', 'از', 'که', 'این', 'آن', 'را', 'با', 'برای',
        'تا', 'کن', 'هم', 'چه', 'اما', 'ولی', 'پس', 'چون', 'اگر', 'تو',
        'بر', 'زیر', 'کنار', 'جلو', 'پشت', 'بین', 'میان', 'نزد',
        'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده',
        'یکی', 'دوتا', 'سه‌تا', 'چندتا', 'چند', 'چندین', 'بعضی', 'همه',
        'شد', 'است', 'هست', 'بود', 'باشد', 'شود', 'می', 'کرد', 'کن', 'کنید',
        'شده', 'بوده', 'دارد', 'دارم', 'داره', 'داری', 'دارید', 'دارند',
        'نیست', 'نبود', 'نباشد', 'نشد', 'نکن',
        'میخوام', 'می‌خوام', 'میخواهم', 'می‌خواهم', 'میخواد', 'می‌خواد',
        'میخوای', 'می‌خوای', 'بخرم', 'بخرید', 'بخر', 'بگیرم', 'بگیر', 'بگیرید',
        'خریدم', 'خریدم', 'خریدن', 'خرید', 'بخریم', 'سفارش', 'تهیه',
        'لطفا', 'لطفاً', 'خواهش', 'خواهشا', 'خواهشاً', 'ممنون', 'متشکر',
        'سلام', 'درود', 'احتراما', 'باسپاس',
        'من', 'تو', 'او', 'ما', 'شما', 'آنها', 'ایشان', 'خود', 'خودم',
        'خودت', 'خودش', 'خودمان', 'خودتان', 'خودشان',
        'چرا', 'کجا', 'کی', 'چطور', 'چگونه', 'چقدر', 'کدام',
        'این', 'آن', 'اون', 'همین', 'همان', 'همون', 'یه', 'یک',
        'ها', 'های', 'هایی', 'هایم', 'هایت', 'هایش', 'هایمان', 'هایتان', 'هایشان',
        'ی', 'ه', 'ای', 'ان', 'ین', 'م', 'ت', 'ش', 'مان', 'تان', 'شان',
        'رو', 'ام', 'ات', 'اش',
        'امروز', 'دیروز', 'فردا', 'الان', 'اکنون', 'حالا', 'هنوز', 'دیگر',
        'باز', 'همیشه', 'هرگز', 'گاهی', 'بعضا', 'اغلب',
        'برای', 'جهت', 'بخاطر', 'واسه', 'واسط',
        'موجود', 'نیاز', 'لازم', 'ضروری', 'مهم', 'اساسی',
        'بله', 'آره', 'نه', 'نخیر', 'باشه', 'باشد', 'حتما', 'حتماً',
        'فقط', 'تنها', 'مثل', 'مانند', 'شبیه', 'همچون', 'نظیر',
        'هر', 'هیچ', 'کسی', 'چیزی', 'جایی', 'وقتی', 'زمانی'
    }

    tokens = tokenizer.tokenize_words(normalized)

    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if len(token) < 2 or token in persian_stopwords:
            continue
        cleaned_tokens.append(token)

    # Stemming
    final_tokens = []
    for token in cleaned_tokens:
        stem = stemmer.convert(token)
        final_tokens.append(token)
        if stem != token:
            final_tokens.append(stem)

    cleaned = ' '.join(final_tokens)
    logger.debug(f"Retrieval cleaned query: {cleaned}")
    return cleaned

def _clean_category_string(cat_string: str) -> str:
    """
    Clean category strings for better embedding similarity.
    Strips English slugs and normalizes Persian text.
    """
    if not cat_string:
        return ""

    parts = cat_string.split()
    cleaned_parts = []
    for part in parts:
        if re.match(r'^[A-Za-z0-9\-]+$', part):
            continue
        cleaned_parts.append(part)

    cleaned_name = " ".join(cleaned_parts)
    if not cleaned_name:
        cleaned_name = cat_string

    return normalizer.normalize(cleaned_name)
