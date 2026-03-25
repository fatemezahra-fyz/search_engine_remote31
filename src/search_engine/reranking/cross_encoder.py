# src/search_engine/reranking/cross_encoder.py

import time
import logging
import numpy as np
from typing import List, Dict
from src.search_engine.config import K_RESULTS

logger = logging.getLogger(__name__)

def cross_encoder_rerank(query: str, hits: List[Dict], reranker, top_k: int = K_RESULTS) -> List[Dict]:
    """
    High-precision re-ranking of candidate hits using a cross-encoder model.
    """
    if not hits:
        return []

    logger.info(f"Re-ranking {len(hits)} hits for query: '{query}'")
    # Richer passage construction
    passages = []
    for h in hits:
        src = h.get('_source', {})
        title = src.get('product_title_fa', '')
        brand = src.get('brand_name_fa', '')
        category = src.get('category_keywords', '')
        passage = f"{title} {brand} {category}".strip()
        passages.append(passage)

    pairs = [(query, p) for p in passages]

    try:
        start = time.time()
        raw_scores = reranker.predict(pairs, convert_to_tensor=False)
        duration = time.time() - start
        logger.info(f"Cross-encoder inference took {duration:.2f}s for {len(hits)} pairs")

        for h, s in zip(hits, raw_scores):
            # REFACTOR: Apply sigmoid normalization
            calibrated_score = 1.0 / (1.0 + np.exp(-float(s)))
            h['rerank_score'] = calibrated_score
            h['final_score'] = calibrated_score

        reranked = sorted(hits, key=lambda h: h.get('rerank_score', 0.0), reverse=True)
        return reranked[:top_k]
    except Exception as e:
        logger.warning(f"⚠️ Reranking failed: {e}")
        return hits[:top_k]
