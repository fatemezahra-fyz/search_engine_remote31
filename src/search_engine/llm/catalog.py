# src/search_engine/llm/catalog.py

import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class CatalogContextProvider:
    def __init__(self, es_client, index_name):
        self.es_client = es_client
        self.index_name = index_name

    def build_catalog_context(self, selected_category: Optional[str] = None) -> Dict[str, List[str]]:
        logger.info(f"Building catalog context. Category filter: {selected_category}")
        category_filter = []
        if selected_category:
            category_filter.append({"term": {"category_keywords": selected_category}})

        try:
            agg_query = {
                "size": 0,
                "query": {"bool": {"filter": category_filter}} if category_filter else {"match_all": {}},
                "aggs": {
                    "categories": {"terms": {"field": "category_keywords", "size": 10}},
                    "attributes": {
                        "nested": {"path": "product_attributes"},
                        "aggs": {
                            "attr_keys": {"terms": {"field": "product_attributes.Key.keyword", "size": 50}}
                        }
                    }
                }
            }

            response = self.es_client.search(index=self.index_name, body=agg_query)

            categories = [c["key"] for c in response.get("aggregations", {}).get("categories", {}).get("buckets", [])]
            attributes = [a["key"] for a in response.get("aggregations", {}).get("attributes", {}).get("attr_keys", {}).get("buckets", [])]

            logger.info(f"Catalog context built: {len(categories)} categories, {len(attributes)} attributes found")
            return {
                "categories": categories[:30],
                "attributes": attributes[:30],
                "brands": []
            }
        except Exception as e:
            logger.error(f"❌ Catalog context build failed: {e}")
            return {"brands": [], "categories": [], "attributes": []}
