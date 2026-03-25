# src/search_engine/multimodal/captioning.py

import logging
import base64
from io import BytesIO
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from src.search_engine.config import LLM_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

class ImageCaptioner: # REFACTOR: uses Ollama multimodal LLM
    """
    REFACTOR: Uses the same Ollama LLM for image captioning instead of a local BLIP model.
    Note: Requires a multimodal model in Ollama (like llava) to work effectively,
    but follows the infrastructure setup of the query enhancer.
    """
    def __init__(self, model_name: str = LLM_MODEL, ollama_base_url: str = OLLAMA_BASE_URL):
        logger.info(f"Initializing ImageCaptioner using Ollama model {model_name} at {ollama_base_url}")
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.2
        )

    def _convert_image_to_base64(self, image_path_or_pil):
        if isinstance(image_path_or_pil, str):
            with open(image_path_or_pil, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            buffered = BytesIO()
            image_path_or_pil.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_caption(self, image_path, prompt="Describe this product in detail for an e-commerce search engine."):
        logger.info("Generating caption using Ollama multimodal interface...")
        try:
            image_base64 = self._convert_image_to_base64(image_path)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            )

            response = self.llm.invoke([message])
            caption = response.content.strip()
            logger.info(f"Ollama generated caption: {caption[:100]}...")
            return caption
        except Exception as e:
            logger.error(f"Image captioning via Ollama failed: {e}")
            return "Error generating caption"
