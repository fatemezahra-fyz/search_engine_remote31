# src/search_engine/core.py

import logging
import re
import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.search_engine.config import *
from src.search_engine.nlp.preprocessing import normalizer, clean_query_for_llm, clean_query_for_retrieval
from src.search_engine.nlp.spelling import Speller, detect_and_convert_keyboard_layout
from src.search_engine.llm.enhancer import LLMQueryEnhancer, CaptionEnhancer
from src.search_engine.llm.catalog import CatalogContextProvider
from src.search_engine.retrieval.engine import search_sparse, search_dense, search_llm_enhanced, CategoryDetector
from src.search_engine.reranking.fusion import reciprocal_rank_fusion
from src.search_engine.reranking.cross_encoder import cross_encoder_rerank
from src.search_engine.personalization.engine import PersonalizationEngine
from src.search_engine.multimodal.captioning import ImageCaptioner

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        logger.info("Initializing SearchEngine core components...")
        start_time = time.time()
        self.es_client = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASSWORD))
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(CROSS_ENCODER_MODEL, device='cpu')

        # REFACTOR: All LLM components use the same Ollama configuration
        self.llm_enhancer = LLMQueryEnhancer(LLM_MODEL, OLLAMA_BASE_URL)
        self.image_captioner = ImageCaptioner(LLM_MODEL, OLLAMA_BASE_URL)
        self.caption_enhancer = CaptionEnhancer(self.llm_enhancer)

        self.catalog_provider = CatalogContextProvider(self.es_client, INDEX_NAME)
        self.category_detector = CategoryDetector(self.es_client, self.embedding_model)
        self.personalization = PersonalizationEngine(normalizer)
        self.speller = Speller(self.es_client)

        logger.info(f"SearchEngine initialized in {time.time() - start_time:.2f}s")

    def classify_query_intent(self, query: str) -> str:
        # REFACTOR: rule-based intent classification
        if not query: return "AMBIGUOUS"
        words = query.split()
        has_model_number = any(re.search(r'[A-Za-z]+\d+|\d+[A-Za-z]+', word) for word in words)
        intent = "PRECISE" if has_model_number else ("AMBIGUOUS" if len(words) <= 2 else "DESCRIPTIVE")
        logger.info(f"Query intent classified as {intent} for: '{query}'")
        return intent

    def process_query_complete(self, query: str) -> dict:
        logger.info(f"Processing query: '{query}'")
        words = query.split()
        converted_words = []
        any_conversion = False
        for word in words:
            converted, was_converted, _ = detect_and_convert_keyboard_layout(word)
            if was_converted: any_conversion = True
            converted_words.append(converted)
        keyboard_fixed = " ".join(converted_words)

        suggestions = self.speller.suggest_corrections(keyboard_fixed)
        spell_corrected = suggestions[0]['text'] if suggestions else keyboard_fixed

        result = {
            'original': query,
            'keyboard_converted': keyboard_fixed,
            'keyboard_conversion_applied': any_conversion,
            'cleaned_llm': clean_query_for_llm(spell_corrected),
            'cleaned_retrieval': clean_query_for_retrieval(spell_corrected),
            'spell_corrected': spell_corrected,
            'spell_correction_applied': spell_corrected != keyboard_fixed,
            'suggestions': suggestions
        }
        logger.info(f"Query processing complete. Final form: '{spell_corrected}'")
        return result

    def unified_search(self, query: str, category: str = None, user_id: str = None, use_personalization: bool = True, catalog_context: dict = None):
        logger.info(f"🚀 Starting unified search for: '{query}' (category: {category}, user: {user_id})")
        start_time = time.time()

        intent = self.classify_query_intent(query)

        logger.info("Executing retrieval methods...")
        sparse_hits = search_sparse(self.es_client, query, category)
        dense_hits = search_dense(self.es_client, self.embedding_model, query, category)
        llm_hits, enhanced_query = search_llm_enhanced(self.es_client, self.llm_enhancer, query, intent, category, catalog_context)
        llm_dense_hits = search_dense(self.es_client, self.embedding_model, enhanced_query, category)

        logger.info(f"Retrieved: sparse={len(sparse_hits)}, dense={len(dense_hits)}, llm={len(llm_hits)}, llm_dense={len(llm_dense_hits)}")

        logger.info("Merging results with RRF...")
        combined_pool = reciprocal_rank_fusion([sparse_hits, dense_hits, llm_hits, llm_dense_hits])

        logger.info(f"Re-ranking {len(combined_pool)} candidates...")
        reranked = cross_encoder_rerank(query, combined_pool, self.reranker, top_k=K_RESULTS * 2)

        if use_personalization and user_id:
            logger.info(f"Applying personalization for user: {user_id}")
            reranked = self.personalization.apply(reranked, user_id)

        final_results = reranked[:K_RESULTS]
        logger.info(f"Unified search completed in {time.time() - start_time:.2f}s, returning {len(final_results)} results")
        return final_results
