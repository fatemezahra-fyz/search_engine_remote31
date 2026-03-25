# src/search_engine/personalization/engine.py

import logging
import pandas as pd
from collections import Counter
from typing import List, Dict

logger = logging.getLogger(__name__)

class PersonalizationEngine:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.user_histories = {}
        self.full_history_df = None

    def load_history(self, csv_path='./user_history.csv'):
        logger.info(f"Loading user history from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
            self.full_history_df = df.set_index(df.columns[0])
            logger.info(f"✅ Loaded history for {len(self.full_history_df)} users")
            return True
        except Exception as e:
            logger.warning(f"⚠️ User history load failed: {e}")
            return False

    def load_user_profile(self, user_id: str):
        logger.info(f"Loading profile for user: {user_id}")
        if self.full_history_df is not None and user_id in self.full_history_df.index:
            row = self.full_history_df.loc[user_id]
            products = str(row.get('product_title_fa_all', '')).split(', ')
            brands = str(row.get('brand_name_fa_all', '')).split(', ')
            categories = str(row.get('category_title_fa_all', '')).split(', ')

            valid_brands = [b.strip() for b in brands if b.strip() and b.strip() != 'متفرقه']
            valid_cats = [c.strip() for c in categories if c.strip()]

            self.user_histories[user_id] = {
                'products': products,
                'top_brands': [b for b, _ in Counter(valid_brands).most_common(5)],
                'top_categories': [c for c, _ in Counter(valid_cats).most_common(5)]
            }
            logger.info(f"Successfully cached profile for user {user_id}")
            return True
        logger.warning(f"User {user_id} not found in history dataframe")
        return False

    def apply(self, hits: List[Dict], user_id: str) -> List[Dict]:
        if not user_id or user_id not in self.user_histories:
            return hits

        logger.info(f"Applying personalization for user {user_id}")
        profile = self.user_histories[user_id]
        pref_brands = {self.normalizer.normalize(b) for b in profile['top_brands']}
        pref_cats = {self.normalizer.normalize(c) for c in profile['top_categories']}
        purchased_products = {self.normalizer.normalize(p) for p in profile['products']}

        for h in hits:
            src = h.get('_source', {})
            title = self.normalizer.normalize(src.get('product_title_fa', ''))
            brand = self.normalizer.normalize(src.get('brand_name_fa', ''))
            category = self.normalizer.normalize(src.get('category_keywords', ''))

            score = h.get('final_score', h.get('_score', 0.0))
            boost = 0.0
            if brand in pref_brands:
                boost += 0.05
                logger.debug(f"Brand match boost (+0.05) for {brand}")
            if category in pref_cats:
                boost += 0.03
                logger.debug(f"Category match boost (+0.03) for {category}")

            multiplier = 1.0
            if title in purchased_products:
                multiplier = 0.85
                logger.debug(f"Purchased item demotion (0.85x) for {title}")

            h['final_score'] = (score + boost) * multiplier

        return sorted(hits, key=lambda x: x.get('final_score', 0.0), reverse=True)
