# src/search_engine/llm/enhancer.py

import logging
import re
from typing import Dict, List, Optional
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.search_engine.config import LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

class LLMQueryEnhancer:
    def __init__(self, model_name: str = LLM_MODEL, ollama_base_url: str = OLLAMA_BASE_URL):
        logger.info(f"Initializing LLMQueryEnhancer with model {model_name} at {ollama_base_url}")
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            num_predict=100,
            base_url=ollama_base_url
        )

        self.prompt = PromptTemplate(
            input_variables=["categories", "attributes", "user_query"],
            template="""شما یک دستیار هوشمند جستجوی محصول در فروشگاه آنلاین دیجی‌کالا هستید.

**اطلاعات دسته‌بندی فعلی:**
دسته‌های موجود: {categories}
ویژگی‌های محصولات: {attributes}

**وظیفه شما:**
کاربر یک توصیف طبیعی از محصول مورد نظرش داده است. شما باید این توصیف را به کلمات کلیدی دقیق و مختصر برای جستجو در پایگاه داده تبدیل کنید.

**قوانین مهم:**
1. فقط کلمات کلیدی مرتبط با محصول بنویسید (نه نام دسته‌بندی)
2. از اطلاعات دسته‌بندی و ویژگی‌ها برای درک بهتر استفاده کنید
3. **تبدیل کنید توصیفات کلی را به ویژگی‌های فنی و مشخصات دقیق:**
   - "گرم" → نوع جنس (پشمی، کشمیری، پشم شترف، کرکی، پلی‌استر ضخیم)
   - "سبک" → نوع جنس (نخی، کتان، ویسکوز، ابریشمی)
   - "خنک" → نوع جنس (نخ، لینن، ویسکوز)
   - "خوب" → برند معتبر، کیفیت بالا
   - "ارزان" → اقتصادی، قیمت مناسب
   - "راحت" → فری‌سایز، کش دار، ارگونومیک
4. از ویژگی‌های موجود در لیست attributes استفاده کنید
5. کلمات فارسی و انگلیسی را ترکیب کنید در صورت نیاز
6. حداکثر 3 عبارت جستجو (هر کدام 2-4 کلمه)
7. عبارات را با کاما از هم جدا کنید
8. هیچ توضیح اضافی ندهید، فقط کلمات کلیدی

**نمونه‌های صحیح:**
مثال: "گوشی خوب برای بازی با صفحه نمایش بزرگ" -> گوشی گیمینگ رم ۸ گیگ, موبایل پردازنده اسنپدراگون, صفحه AMOLED شش اینچ

**حالا نوبت شماست:**
توصیف کاربر: {user_query}

کلمات کلیدی جستجو:"""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def enhance_query(self, user_query: str, catalog_context: Dict[str, List[str]], intent: str) -> str:
        if intent == "PRECISE":
            logger.info(f"Skipping enhancement for PRECISE intent: {user_query}")
            return user_query

        try:
            logger.info(f"Requesting LLM enhancement for: {user_query}")
            result = self.chain.invoke({
                "categories": ", ".join(catalog_context.get("categories", [])),
                "attributes": ", ".join(catalog_context.get("attributes", [])),
                "user_query": user_query
            })

            raw_output = result['text'].strip()

            # REFACTOR: Strip common LLM preamble patterns from the output using regex
            noise_patterns = [
                r'^کلمات کلیدی جستجو[:：]\s*', r'^Query[:：]\s*', r'^خروجی[:：]\s*', r'^عبارت جستجو[:：]\s*',
                r'^\s*[*]+\s*', r'^\s*[-]+\s*',
            ]

            enhanced = raw_output
            for pattern in noise_patterns:
                enhanced = re.sub(pattern, '', enhanced, flags=re.IGNORECASE | re.MULTILINE)

            enhanced = enhanced.strip()
            logger.info(f"LLM enhanced query: '{user_query}' -> '{enhanced}'")
            return enhanced
        except Exception as e:
            logger.error(f"❌ LLM enhancement failed: {e}")
            return user_query

class CaptionEnhancer:
    def __init__(self, llm_enhancer):
        self.llm = llm_enhancer

    def enhance_caption(self, english_caption: str) -> str:
        """
        REFACTOR: Convert English caption to Persian search query with few-shot examples and strict constraints.
        """
        logger.info(f"Enhancing image caption: {english_caption}")
        prompt = f"""
You are a product search expert for a Persian e-commerce platform.

Task: Convert the provided English image caption into a precise Persian search query.

Rules:
1. Use ONLY Persian product terminology.
2. Max 3 comma-separated terms.
3. No category names, no explanations.
4. Extract key attributes (color, material, brand, features).

Examples:
- "red leather handbag with gold chain" -> کیف دستی چرم قرمز, کیف زنجیر طلایی, کیف مجلسی زنانه
- "blue running shoes with white soles" -> کفش رانینگ آبی, کتانی ورزشی راحت, کفش دویدن کفه سفید
- "black mechanical gaming keyboard" -> کیبورد گیمینگ مکانیکی, کیبورد مخصوص بازی مشکی
- "stainless steel water bottle 500ml" -> فلاسک استیل نیم لیتری, قمقمه کوهنوردی فلزی
- "floral summer cotton dress for girls" -> پیراهن نخی گلدار دخترانه, لباس تابستانی خنک

English Caption: {english_caption}

Return ONLY the Persian search query.

Persian Query:"""
        try:
            response = self.llm.llm.invoke(prompt)
            enhanced = response.content.strip()
            logger.info(f"Enhanced image caption: '{english_caption}' -> '{enhanced}'")
            return enhanced
        except Exception as e:
            logger.error(f"Image caption enhancement failed: {e}")
            return ""

    def enhance_with_context(self, caption, category_hint=None):
        """
        REFACTOR: Enhanced version with category context and strict structural constraints.
        """
        logger.info(f"Enhancing image caption with context: {caption} (hint: {category_hint})")
        prompt = f"""
You are a product search expert for a Persian e-commerce platform.

Image Description (English): {caption}
{"Category Hint: " + category_hint if category_hint else ""}

Generate a Persian search query with:
1. Product type (محصول)
2. Brand if visible (برند)
3. Key features (ویژگی‌ها)
4. Color/material (رنگ/جنس)

Rules:
- Output language: Persian.
- Max 3 comma-separated terms.
- No explanations or preamble.

Format: [product] [brand] [features] [color/material]
Example: کفش ورزشی نایک رانینگ مشکی, کتانی پیاده روی منعطف

Persian Query:"""
        try:
            response = self.llm.llm.invoke(prompt)
            enhanced = response.content.strip()
            logger.info(f"Enhanced image caption with context: '{enhanced}'")
            return enhanced
        except Exception as e:
            logger.error(f"Image caption enhancement with context failed: {e}")
            return ""
