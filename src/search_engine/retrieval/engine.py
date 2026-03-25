# src/search_engine/retrieval/engine.py

import logging
import time
from typing import List, Dict, Optional, Tuple
from src.search_engine.config import INDEX_NAME, RETRIEVER_CANDIDATES, CATEGORY_DETECT_TOP_N, CATEGORY_SIM_WEIGHT, CATEGORY_CONFIDENCE_THRESHOLD
from src.search_engine.utils import cosine_sim

logger = logging.getLogger(__name__)

def search_sparse(es_client, query: str, category: Optional[str] = None, k: int = RETRIEVER_CANDIDATES) -> List[Dict]:
    logger.info(f"Running sparse search for: '{query}' (category: {category})")
    query_words = query.split()
    should_clauses = []
    for i, word in enumerate(query_words):
        position_boost = max(1.0, 3.0 - (i * 0.5))
        should_clauses.append({"match": {"product_title_fa": {"query": word, "boost": position_boost * 2.0, "fuzziness": "AUTO"}}})
        should_clauses.append({"match": {"brand_name_fa": {"query": word, "boost": position_boost * 1.0}}})
        should_clauses.append({"match": {"category_keywords": {"query": word, "boost": position_boost * 1.0}}})

    should_clauses.append({"match_phrase": {"product_title_fa": {"query": query, "boost": 5.0}}})

    body = {
        "size": k,
        "query": {"bool": {"should": should_clauses, "minimum_should_match": 1, "filter": []}},
        "_source": ["product_title_fa", "brand_name_fa", "category_keywords", "price", "shop", "url_code"]
    }
    if category:
        body["query"]["bool"]["filter"].append({"term": {"category_keywords": category}})

    try:
        res = es_client.search(index=INDEX_NAME, body=body)
        hits = res['hits']['hits']
        logger.info(f"Sparse search returned {len(hits)} results")
        return hits
    except Exception as e:
        logger.warning(f"⚠️ Sparse search failed: {e}")
        return []

def search_dense # REFACTOR: uses query: prefix for e5(es_client, embedding_model, query: str, category: Optional[str] = None, k: int = RETRIEVER_CANDIDATES) -> List[Dict]:
    logger.info(f"Running dense search for: '{query}' (category: {category})")
    try:
        vector = embedding_model.encode("query: " + query).tolist()
        body = {
            "size": k,
            "knn": {"field": "product_title_fa_vector", "query_vector": vector, "k": k, "num_candidates": k}
        }
        if category:
            body["knn"]["filter"] = {"term": {"category_keywords": category}}

        res = es_client.search(index=INDEX_NAME, body=body)
        hits = res['hits']['hits']
        logger.info(f"Dense search returned {len(hits)} results")
        return hits
    except Exception as e:
        logger.warning(f"⚠️ Dense search failed: {e}")
        return []

def search_llm_enhanced(es_client, llm_enhancer, query: str, intent: str, category: Optional[str] = None, catalog_context: Optional[Dict] = None, k: int = RETRIEVER_CANDIDATES) -> Tuple[List[Dict], str]:
    logger.info(f"Running LLM enhanced search for: '{query}'")
    if not llm_enhancer or not catalog_context:
        logger.warning("LLM enhancer or catalog context missing, skipping enhanced search")
        return [], query

    enhanced_query = llm_enhancer.enhance_query(query, catalog_context, intent)
    search_terms = [term.strip() for term in enhanced_query.split(",")]

    should_clauses = []
    for term in search_terms:
        should_clauses.append({
            "multi_match": {
                "query": term,
                "fields": ["product_title_fa^3", "brand_name_fa^2", "category_keywords^2"],
                "type": "cross_fields",
                "operator": "and",
                "boost": 2.0
            }
        })
    should_clauses.append({"match_phrase": {"product_title_fa": {"query": enhanced_query.replace(",", " "), "boost": 5.0}}})

    body = {
        "size": k,
        "query": {"bool": {"should": should_clauses, "minimum_should_match": 1, "filter": []}},
        "_source": ["product_title_fa", "brand_name_fa", "category_keywords", "price", "shop", "url_code"]
    }
    if category:
        body["query"]["bool"]["filter"].append({"term": {"category_keywords": category}})

    try:
        res = es_client.search(index=INDEX_NAME, body=body)
        hits = res['hits']['hits']
        logger.info(f"LLM enhanced search returned {len(hits)} results")
        return hits, enhanced_query
    except Exception as e:
        logger.warning(f"⚠️ LLM search failed: {e}")
        return [], enhanced_query

class CategoryDetector:
    def __init__(self, es_client, embedding_model):
        self.es_client = es_client
        self.embedding_model = embedding_model
        self.cache = {}
        self.cleaned_map = {}
        self.doc_counts = {}
        self.last_refresh = 0.0

    def refresh_cache(self):
        logger.info("Refreshing category detector cache...")
        try:
            body = {"size": 0, "aggs": {"all_categories": {"terms": {"field": "category_keywords", "size": 1000}}}}
            res = self.es_client.search(index=INDEX_NAME, body=body)
            buckets = res.get("aggregations", {}).get("all_categories", {}).get("buckets", [])
            if buckets:
                from src.search_engine.nlp.preprocessing import _clean_category_string
                cats = [b["key"] for b in buckets]
                self.doc_counts = {b["key"]: b["doc_count"] for b in buckets}
                cleaned_cats = [_clean_category_string(c) for c in cats]
                self.cleaned_map = dict(zip(cleaned_cats, cats))
                prefixed_cats = ["passage: " + c for c in cleaned_cats]
                vectors = self.embedding_model.encode(prefixed_cats)
                for cat, vec in zip(cats, vectors):
                    self.cache[cat] = vec.tolist()
                self.last_refresh = time.time()
                logger.info(f"Category cache refreshed with {len(self.cache)} categories")
                return True
            else:
                logger.warning("No categories found to cache")
                return False
        except Exception as e:
            logger.error(f"❌ Category cache refresh failed: {e}")
            return False

    def detect(self, query_text: str, top_n: int = CATEGORY_DETECT_TOP_N) -> List[Tuple]:
        logger.info(f"Detecting categories for query: '{query_text}'")
        if time.time() - self.last_refresh > 3600 or not self.cache:
            self.refresh_cache()

        try:
            body = {
                "size": 0,
                "query": {"multi_match": {"query": query_text, "fields": ["product_title_fa^2", "brand_name_fa", "category_keywords"], "operator": "or", "minimum_should_match": "50%"}},
                "aggs": {"top_categories": {"terms": {"field": "category_keywords", "size": top_n * 5}}}
            }
            res = self.es_client.search(index=INDEX_NAME, body=body)
            buckets = res.get("aggregations", {}).get("top_categories", {}).get("buckets", [])
            agg_map = {b["key"]: b["doc_count"] for b in buckets}
            logger.debug(f"Aggregation matched {len(agg_map)} categories")
        except Exception as e:
            logger.warning(f"Aggregation failed: {e}")
            agg_map = {}

        q_vec = self.embedding_model.encode("query: " + query_text)
        scored = []
        max_total_count = max(self.doc_counts.values()) if self.doc_counts else 1

        for cat, vec in self.cache.items():
            count = agg_map.get(cat, 0)
            norm_count = float(count) # REFACTOR: stable normalization against total cat size / self.doc_counts.get(cat, max_total_count) if self.doc_counts.get(cat, 0) > 0 else 0.0
            sim = cosine_sim(q_vec, vec)
            combined = CATEGORY_SIM_WEIGHT * sim + (1 - CATEGORY_SIM_WEIGHT) * norm_count
            if combined >= CATEGORY_CONFIDENCE_THRESHOLD: # REFACTOR: confidence fallback
                scored.append((cat, combined, count, sim))

        results = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
        logger.info(f"Detected {len(results)} categories above threshold")
        return results
