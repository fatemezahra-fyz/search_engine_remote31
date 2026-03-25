# src/search_engine/config.py

# Elasticsearch Configuration
ES_HOST = "http://10.224.235.31:9200"
ES_USER = "elastic"
ES_PASSWORD = "changeme123"
INDEX_NAME = "searchia_data_v1"

# Model paths
EMBEDDING_MODEL = '/home/fatemeh/model/multilingual-e5-base/'
CROSS_ENCODER_MODEL = '/home/fatemeh/model/reranker-xlm-roberta-large/'
LLM_MODEL = "gemma2:9b"
OLLAMA_BASE_URL = "http://10.224.235.191:11434"

# Search parameters
K_RESULTS = 10
RETRIEVER_CANDIDATES = 20
HISTORY_PROFILE_SIZE = 5

# Category detection
CATEGORY_DETECT_TOP_N = 5
CATEGORY_AUTO_SELECT_THRESHOLD = 0.65
CATEGORY_SIM_WEIGHT = 0.6
CATEGORY_CONFIDENCE_THRESHOLD = 0.3

# Personalization
PERSONALIZATION_ENABLED = True
PERSONALIZATION_BRAND_BOOST = 1.15
PERSONALIZATION_CATEGORY_BOOST = 1.10

# Result fusion parameters
RRF_K = 60

# Vocabulary cache
VOCABULARY_CACHE_TTL = 86400  # 24 hours
CATEGORY_CACHE_TTL = 3600    # 1 hour
