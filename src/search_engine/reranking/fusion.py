# src/search_engine/reranking/fusion.py

import logging
from typing import List, Dict
from src.search_engine.config import RRF_K

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion # REFACTOR: RRF implementation(hits_list: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
    """
    logger.info(f"Fusing {len(hits_list)} result sets using RRF (k={k})")
    scores = {} # doc_id -> rrf_score
    all_docs = {}

    for i, hits in enumerate(hits_list):
        logger.debug(f"Processing list {i} with {len(hits)} hits")
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit.get('_id') or hit.get('id')
            all_docs[doc_id] = hit
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for doc_id in sorted_ids:
        hit = all_docs[doc_id]
        hit['rrf_score'] = scores[doc_id]
        hit['_score'] = scores[doc_id]
        fused.append(hit)

    logger.info(f"RRF fusion complete. Total unique documents: {len(fused)}")
    return fused
