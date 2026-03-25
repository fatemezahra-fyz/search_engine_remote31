# Unified Hybrid Search Engine

A Persian/Farsi e-commerce search engine combining Elasticsearch (BM25 + kNN) with LLM-enhanced query expansion and cross-encoder re-ranking.

## Features

- **Unified Hybrid Search**: Combines sparse (BM25), dense (kNN), and LLM-enhanced retrieval.
- **Multimodal Search**: Supports both text queries and image uploads (using BLIP for captioning).
- **Advanced Category Detection**: Hybrid aggregation and embedding-based category selection.
- **Query Enhancement**: Rule-based intent classification and LLM expansion (Gemma2:9b via Ollama).
- **Intelligent Re-ranking**: XLM-RoBERTa cross-encoder for high-precision ranking.
- **Reciprocal Rank Fusion (RRF)**: Robust merging of results from multiple retrieval strategies.
- **Personalization**: User history-based boosting and demotion.
- **Robust Persian NLP**: Keyboard layout fix, normalization, spell checking, and stemming.

## Setup

### Prerequisites

- Python 3.8+
- Elasticsearch 8.x
- Ollama (running Gemma2:9b)

### Running Locally (Without Docker)

1. Clone the repository.
2. Install dependencies:
   Collecting elasticsearch (from -r requirements.txt (line 1))
  Downloading elasticsearch-9.4.0-py3-none-any.whl.metadata (9.0 kB)
Collecting sentence-transformers (from -r requirements.txt (line 2))
  Downloading sentence_transformers-5.5.0-py3-none-any.whl.metadata (18 kB)
Collecting langchain (from -r requirements.txt (line 3))
  Downloading langchain-1.3.1-py3-none-any.whl.metadata (5.8 kB)
Collecting langchain-community (from -r requirements.txt (line 4))
  Downloading langchain_community-0.4.1-py3-none-any.whl.metadata (3.0 kB)
Collecting flask (from -r requirements.txt (line 5))
  Downloading flask-3.1.3-py3-none-any.whl.metadata (3.2 kB)
Collecting flask-cors (from -r requirements.txt (line 6))
  Downloading flask_cors-6.0.2-py3-none-any.whl.metadata (5.3 kB)
Collecting pandas (from -r requirements.txt (line 7))
  Downloading pandas-3.0.3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (79 kB)
Collecting numpy (from -r requirements.txt (line 8))
  Downloading numpy-2.4.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (6.6 kB)
Collecting arabic-reshaper (from -r requirements.txt (line 9))
  Downloading arabic_reshaper-3.0.1-py3-none-any.whl.metadata (13 kB)
Collecting python-bidi (from -r requirements.txt (line 10))
  Downloading python_bidi-0.6.10-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.3 kB)
Collecting parsivar (from -r requirements.txt (line 11))
  Downloading parsivar-0.2.3.1-py3-none-any.whl.metadata (242 bytes)
Collecting transformers (from -r requirements.txt (line 12))
  Downloading transformers-5.8.1-py3-none-any.whl.metadata (33 kB)
Collecting pillow (from -r requirements.txt (line 13))
  Downloading pillow-12.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting torch (from -r requirements.txt (line 14))
  Downloading torch-2.12.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (31 kB)
Collecting torchvision (from -r requirements.txt (line 15))
  Downloading torchvision-0.27.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting torchaudio (from -r requirements.txt (line 16))
  Downloading torchaudio-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (6.9 kB)
Collecting Ollama (from -r requirements.txt (line 17))
  Downloading ollama-0.6.2-py3-none-any.whl.metadata (5.8 kB)
Collecting anyio (from elasticsearch->-r requirements.txt (line 1))
  Downloading anyio-4.13.0-py3-none-any.whl.metadata (4.5 kB)
Collecting elastic-transport<10,>=9.2.0 (from elasticsearch->-r requirements.txt (line 1))
  Downloading elastic_transport-9.4.0-py3-none-any.whl.metadata (3.9 kB)
Collecting python-dateutil (from elasticsearch->-r requirements.txt (line 1))
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting sniffio (from elasticsearch->-r requirements.txt (line 1))
  Downloading sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
Requirement already satisfied: typing-extensions in /home/jules/.pyenv/versions/3.12.13/lib/python3.12/site-packages (from elasticsearch->-r requirements.txt (line 1)) (4.15.0)
Collecting urllib3<3,>=2 (from elastic-transport<10,>=9.2.0->elasticsearch->-r requirements.txt (line 1))
  Downloading urllib3-2.7.0-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi (from elastic-transport<10,>=9.2.0->elasticsearch->-r requirements.txt (line 1))
  Downloading certifi-2026.4.22-py3-none-any.whl.metadata (2.5 kB)
Collecting huggingface-hub>=0.23.0 (from sentence-transformers->-r requirements.txt (line 2))
  Downloading huggingface_hub-1.15.0-py3-none-any.whl.metadata (14 kB)
Collecting scikit-learn>=0.22.0 (from sentence-transformers->-r requirements.txt (line 2))
  Downloading scikit_learn-1.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (11 kB)
Collecting scipy>=1.0.0 (from sentence-transformers->-r requirements.txt (line 2))
  Downloading scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting tqdm>=4.0.0 (from sentence-transformers->-r requirements.txt (line 2))
  Downloading tqdm-4.67.3-py3-none-any.whl.metadata (57 kB)
Collecting packaging>=20.0 (from transformers->-r requirements.txt (line 12))
  Downloading packaging-26.2-py3-none-any.whl.metadata (3.5 kB)
Collecting pyyaml>=5.1 (from transformers->-r requirements.txt (line 12))
  Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting regex>=2025.10.22 (from transformers->-r requirements.txt (line 12))
  Downloading regex-2026.5.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers->-r requirements.txt (line 12))
  Downloading tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
Collecting typer (from transformers->-r requirements.txt (line 12))
  Downloading typer-0.25.1-py3-none-any.whl.metadata (15 kB)
Collecting safetensors>=0.4.3 (from transformers->-r requirements.txt (line 12))
  Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Collecting filelock>=3.10.0 (from huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Downloading filelock-3.29.0-py3-none-any.whl.metadata (2.0 kB)
Collecting fsspec>=2023.5.0 (from huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Downloading fsspec-2026.4.0-py3-none-any.whl.metadata (10 kB)
Collecting hf-xet<2.0.0,>=1.4.3 (from huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Downloading hf_xet-1.5.0-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (4.9 kB)
Collecting httpx<1,>=0.23.0 (from huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting httpcore==1.* (from httpx<1,>=0.23.0->huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Using cached httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting idna (from httpx<1,>=0.23.0->huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Downloading idna-3.15-py3-none-any.whl.metadata (7.7 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.23.0->sentence-transformers->-r requirements.txt (line 2))
  Using cached h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting langchain-core<2.0.0,>=1.4.0 (from langchain->-r requirements.txt (line 3))
  Downloading langchain_core-1.4.0-py3-none-any.whl.metadata (4.5 kB)
Collecting langgraph<1.3.0,>=1.2.0 (from langchain->-r requirements.txt (line 3))
  Downloading langgraph-1.2.0-py3-none-any.whl.metadata (8.0 kB)
Collecting pydantic<3.0.0,>=2.7.4 (from langchain->-r requirements.txt (line 3))
  Downloading pydantic-2.13.4-py3-none-any.whl.metadata (109 kB)
Collecting jsonpatch<2.0.0,>=1.33.0 (from langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading jsonpatch-1.33-py2.py3-none-any.whl.metadata (3.0 kB)
Collecting langchain-protocol>=0.0.14 (from langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading langchain_protocol-0.0.15-py3-none-any.whl.metadata (2.4 kB)
Collecting langsmith<1.0.0,>=0.3.45 (from langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading langsmith-0.8.5-py3-none-any.whl.metadata (15 kB)
Collecting tenacity!=8.4.0,<10.0.0,>=8.1.0 (from langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading tenacity-9.1.4-py3-none-any.whl.metadata (1.2 kB)
Collecting uuid-utils<1.0,>=0.12.0 (from langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading uuid_utils-0.15.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.5 kB)
Collecting jsonpointer>=1.9 (from jsonpatch<2.0.0,>=1.33.0->langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading jsonpointer-3.1.1-py3-none-any.whl.metadata (2.4 kB)
Collecting langgraph-checkpoint<5.0.0,>=4.1.0 (from langgraph<1.3.0,>=1.2.0->langchain->-r requirements.txt (line 3))
  Downloading langgraph_checkpoint-4.1.0-py3-none-any.whl.metadata (5.2 kB)
Collecting langgraph-prebuilt<1.2.0,>=1.1.0 (from langgraph<1.3.0,>=1.2.0->langchain->-r requirements.txt (line 3))
  Downloading langgraph_prebuilt-1.1.0-py3-none-any.whl.metadata (5.2 kB)
Collecting langgraph-sdk<0.4.0,>=0.3.0 (from langgraph<1.3.0,>=1.2.0->langchain->-r requirements.txt (line 3))
  Downloading langgraph_sdk-0.3.14-py3-none-any.whl.metadata (1.7 kB)
Collecting xxhash>=3.5.0 (from langgraph<1.3.0,>=1.2.0->langchain->-r requirements.txt (line 3))
  Downloading xxhash-3.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (13 kB)
Collecting ormsgpack>=1.12.0 (from langgraph-checkpoint<5.0.0,>=4.1.0->langgraph<1.3.0,>=1.2.0->langchain->-r requirements.txt (line 3))
  Downloading ormsgpack-1.12.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
Collecting orjson>=3.11.5 (from langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.3.0,>=1.2.0->langchain->-r requirements.txt (line 3))
  Downloading orjson-3.11.9-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (41 kB)
Collecting requests-toolbelt>=1.0.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Using cached requests_toolbelt-1.0.0-py2.py3-none-any.whl.metadata (14 kB)
Collecting requests>=2.0.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading requests-2.34.2-py3-none-any.whl.metadata (4.8 kB)
Collecting zstandard>=0.23.0 (from langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Using cached zstandard-0.25.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (3.3 kB)
Collecting annotated-types>=0.6.0 (from pydantic<3.0.0,>=2.7.4->langchain->-r requirements.txt (line 3))
  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.46.4 (from pydantic<3.0.0,>=2.7.4->langchain->-r requirements.txt (line 3))
  Downloading pydantic_core-2.46.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Collecting typing-inspection>=0.4.2 (from pydantic<3.0.0,>=2.7.4->langchain->-r requirements.txt (line 3))
  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting langchain-classic<2.0.0,>=1.0.0 (from langchain-community->-r requirements.txt (line 4))
  Downloading langchain_classic-1.0.7-py3-none-any.whl.metadata (5.1 kB)
Collecting SQLAlchemy<3.0.0,>=1.4.0 (from langchain-community->-r requirements.txt (line 4))
  Downloading sqlalchemy-2.0.49-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (9.5 kB)
Collecting aiohttp<4.0.0,>=3.8.3 (from langchain-community->-r requirements.txt (line 4))
  Downloading aiohttp-3.13.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
Collecting dataclasses-json<0.7.0,>=0.6.7 (from langchain-community->-r requirements.txt (line 4))
  Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
Collecting pydantic-settings<3.0.0,>=2.10.1 (from langchain-community->-r requirements.txt (line 4))
  Downloading pydantic_settings-2.14.1-py3-none-any.whl.metadata (3.4 kB)
Collecting httpx-sse<1.0.0,>=0.4.0 (from langchain-community->-r requirements.txt (line 4))
  Downloading httpx_sse-0.4.3-py3-none-any.whl.metadata (9.7 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting attrs>=17.3.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading attrs-26.1.0-py3-none-any.whl.metadata (8.8 kB)
Collecting frozenlist>=1.1.1 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (20 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading multidict-6.7.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.3 kB)
Collecting propcache>=0.2.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading propcache-0.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (16 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community->-r requirements.txt (line 4))
  Downloading yarl-1.23.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (79 kB)
Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7.0,>=0.6.7->langchain-community->-r requirements.txt (line 4))
  Downloading marshmallow-3.26.2-py3-none-any.whl.metadata (7.3 kB)
Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json<0.7.0,>=0.6.7->langchain-community->-r requirements.txt (line 4))
  Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)
Collecting langchain-text-splitters<2.0.0,>=1.1.2 (from langchain-classic<2.0.0,>=1.0.0->langchain-community->-r requirements.txt (line 4))
  Downloading langchain_text_splitters-1.1.2-py3-none-any.whl.metadata (3.3 kB)
Collecting python-dotenv>=0.21.0 (from pydantic-settings<3.0.0,>=2.10.1->langchain-community->-r requirements.txt (line 4))
  Downloading python_dotenv-1.2.2-py3-none-any.whl.metadata (27 kB)
Collecting charset_normalizer<4,>=2 (from requests>=2.0.0->langsmith<1.0.0,>=0.3.45->langchain-core<2.0.0,>=1.4.0->langchain->-r requirements.txt (line 3))
  Downloading charset_normalizer-3.4.7-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Requirement already satisfied: greenlet>=1 in /home/jules/.pyenv/versions/3.12.13/lib/python3.12/site-packages (from SQLAlchemy<3.0.0,>=1.4.0->langchain-community->-r requirements.txt (line 4)) (3.3.2)
Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7.0,>=0.6.7->langchain-community->-r requirements.txt (line 4))
  Using cached mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)
Collecting blinker>=1.9.0 (from flask->-r requirements.txt (line 5))
  Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
Collecting click>=8.1.3 (from flask->-r requirements.txt (line 5))
  Downloading click-8.4.0-py3-none-any.whl.metadata (2.6 kB)
Collecting itsdangerous>=2.2.0 (from flask->-r requirements.txt (line 5))
  Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
Collecting jinja2>=3.1.2 (from flask->-r requirements.txt (line 5))
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting markupsafe>=2.1.1 (from flask->-r requirements.txt (line 5))
  Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting werkzeug>=3.1.0 (from flask->-r requirements.txt (line 5))
  Downloading werkzeug-3.1.8-py3-none-any.whl.metadata (4.0 kB)
Collecting nltk>=3.6.6 (from parsivar->-r requirements.txt (line 11))
  Downloading nltk-3.9.4-py3-none-any.whl.metadata (3.2 kB)
Collecting setuptools<82 (from torch->-r requirements.txt (line 14))
  Downloading setuptools-81.0.0-py3-none-any.whl.metadata (6.6 kB)
Collecting sympy>=1.13.3 (from torch->-r requirements.txt (line 14))
  Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx>=2.5.1 (from torch->-r requirements.txt (line 14))
  Downloading networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
Collecting cuda-toolkit==13.0.2 (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading cuda_toolkit-13.0.2-py2.py3-none-any.whl.metadata (9.4 kB)
Collecting nvidia-cublas<=13.1.1.3,>=13.1.0.3 (from torch->-r requirements.txt (line 14))
  Downloading nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting cuda-bindings<14,>=13.0.3 (from torch->-r requirements.txt (line 14))
  Downloading cuda_bindings-13.2.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (2.3 kB)
Collecting nvidia-cudnn-cu13==9.20.0.48 (from torch->-r requirements.txt (line 14))
  Downloading nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl.metadata (1.9 kB)
Collecting nvidia-cusparselt-cu13==0.8.1 (from torch->-r requirements.txt (line 14))
  Downloading nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl.metadata (12 kB)
Collecting nvidia-nccl-cu13==2.29.7 (from torch->-r requirements.txt (line 14))
  Downloading nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-nvshmem-cu13==3.4.5 (from torch->-r requirements.txt (line 14))
  Downloading nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Collecting triton==3.7.0 (from torch->-r requirements.txt (line 14))
  Downloading triton-3.7.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime==13.0.96.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cuda_runtime-13.0.96-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft==12.0.0.61.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cufile==1.15.1.6.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cufile-1.15.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti==13.0.85.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cuda_cupti-13.0.85-py3-none-manylinux_2_25_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand==10.4.0.35.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver==12.0.4.66.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cusolver-12.0.4.66-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse==12.6.3.3.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cusparse-12.6.3.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink==13.0.88.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_nvjitlink-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-nvrtc==13.0.88.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_cuda_nvrtc-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-nvtx==13.0.85.* (from cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_system == "Linux"->torch->-r requirements.txt (line 14))
  Downloading nvidia_nvtx-13.0.85-py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl.metadata (1.8 kB)
Collecting cuda-pathfinder~=1.1 (from cuda-bindings<14,>=13.0.3->torch->-r requirements.txt (line 14))
  Downloading cuda_pathfinder-1.5.4-py3-none-any.whl.metadata (1.9 kB)
Collecting joblib (from nltk>=3.6.6->parsivar->-r requirements.txt (line 11))
  Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting six>=1.5 (from python-dateutil->elasticsearch->-r requirements.txt (line 1))
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting threadpoolctl>=3.2.0 (from scikit-learn>=0.22.0->sentence-transformers->-r requirements.txt (line 2))
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch->-r requirements.txt (line 14))
  Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting shellingham>=1.3.0 (from typer->transformers->-r requirements.txt (line 12))
  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Collecting rich>=13.8.0 (from typer->transformers->-r requirements.txt (line 12))
  Downloading rich-15.0.0-py3-none-any.whl.metadata (18 kB)
Collecting annotated-doc>=0.0.2 (from typer->transformers->-r requirements.txt (line 12))
  Downloading annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=13.8.0->typer->transformers->-r requirements.txt (line 12))
  Downloading markdown_it_py-4.2.0-py3-none-any.whl.metadata (7.4 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich>=13.8.0->typer->transformers->-r requirements.txt (line 12))
  Downloading pygments-2.20.0-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=13.8.0->typer->transformers->-r requirements.txt (line 12))
  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading elasticsearch-9.4.0-py3-none-any.whl (992 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 992.7/992.7 kB 21.1 MB/s  0:00:00
Downloading elastic_transport-9.4.0-py3-none-any.whl (65 kB)
Downloading urllib3-2.7.0-py3-none-any.whl (131 kB)
Downloading sentence_transformers-5.5.0-py3-none-any.whl (588 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 588.7/588.7 kB 16.6 MB/s  0:00:00
Downloading transformers-5.8.1-py3-none-any.whl (10.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.6/10.6 MB 26.0 MB/s  0:00:00
Downloading huggingface_hub-1.15.0-py3-none-any.whl (663 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 663.6/663.6 kB 42.3 MB/s  0:00:00
Downloading hf_xet-1.5.0-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 54.9 MB/s  0:00:00
Using cached httpx-0.28.1-py3-none-any.whl (73 kB)
Using cached httpcore-1.0.9-py3-none-any.whl (78 kB)
Downloading tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 38.3 MB/s  0:00:00
Downloading langchain-1.3.1-py3-none-any.whl (114 kB)
Downloading langchain_core-1.4.0-py3-none-any.whl (548 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 548.1/548.1 kB 35.4 MB/s  0:00:00
Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
Downloading langgraph-1.2.0-py3-none-any.whl (234 kB)
Downloading langgraph_checkpoint-4.1.0-py3-none-any.whl (56 kB)
Downloading langgraph_prebuilt-1.1.0-py3-none-any.whl (41 kB)
Downloading langgraph_sdk-0.3.14-py3-none-any.whl (97 kB)
Downloading langsmith-0.8.5-py3-none-any.whl (399 kB)
Downloading pydantic-2.13.4-py3-none-any.whl (472 kB)
Downloading pydantic_core-2.46.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 27.1 MB/s  0:00:00
Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (807 kB)
Downloading tenacity-9.1.4-py3-none-any.whl (28 kB)
Downloading uuid_utils-0.15.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (324 kB)
Downloading langchain_community-0.4.1-py3-none-any.whl (2.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 27.1 MB/s  0:00:00
Downloading aiohttp-3.13.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 38.4 MB/s  0:00:00
Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
Downloading httpx_sse-0.4.3-py3-none-any.whl (9.0 kB)
Downloading langchain_classic-1.0.7-py3-none-any.whl (1.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 32.0 MB/s  0:00:00
Downloading langchain_text_splitters-1.1.2-py3-none-any.whl (35 kB)
Downloading marshmallow-3.26.2-py3-none-any.whl (50 kB)
Downloading multidict-6.7.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (256 kB)
Downloading pydantic_settings-2.14.1-py3-none-any.whl (60 kB)
Downloading requests-2.34.2-py3-none-any.whl (73 kB)
Downloading charset_normalizer-3.4.7-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (216 kB)
Downloading idna-3.15-py3-none-any.whl (72 kB)
Downloading sqlalchemy-2.0.49-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (3.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 27.1 MB/s  0:00:00
Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
Downloading yarl-1.23.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (100 kB)
Downloading flask-3.1.3-py3-none-any.whl (103 kB)
Downloading flask_cors-6.0.2-py3-none-any.whl (13 kB)
Downloading pandas-3.0.3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (10.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.9/10.9 MB 31.8 MB/s  0:00:00
Downloading numpy-2.4.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 32.7 MB/s  0:00:00
Downloading arabic_reshaper-3.0.1-py3-none-any.whl (20 kB)
Downloading python_bidi-0.6.10-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (299 kB)
Downloading parsivar-0.2.3.1-py3-none-any.whl (18.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.0/18.0 MB 31.4 MB/s  0:00:00
Downloading pillow-12.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.1/7.1 MB 76.5 MB/s  0:00:00
Downloading torch-2.12.0-cp312-cp312-manylinux_2_28_x86_64.whl (532.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 532.3/532.3 MB 21.5 MB/s  0:00:18
Downloading cuda_toolkit-13.0.2-py2.py3-none-any.whl (2.4 kB)
Downloading nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl (366.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 366.2/366.2 MB 34.5 MB/s  0:00:08
Downloading nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl (170.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.1/170.1 MB 25.1 MB/s  0:00:06
Downloading nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl (206.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 206.0/206.0 MB 17.9 MB/s  0:00:11
Downloading nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (60.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.4/60.4 MB 29.0 MB/s  0:00:02
Downloading triton-3.7.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (201.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.5/201.5 MB 16.9 MB/s  0:00:11
Downloading cuda_bindings-13.2.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 65.7 MB/s  0:00:00
Downloading cuda_pathfinder-1.5.4-py3-none-any.whl (51 kB)
Downloading nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl (423.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 423.1/423.1 MB 13.9 MB/s  0:00:25
Downloading nvidia_cuda_cupti-13.0.85-py3-none-manylinux_2_25_x86_64.whl (10.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.7/10.7 MB 89.1 MB/s  0:00:00
Downloading nvidia_cuda_nvrtc-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (90.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 90.2/90.2 MB 22.3 MB/s  0:00:04
Downloading nvidia_cuda_runtime-13.0.96-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (2.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 58.7 MB/s  0:00:00
Downloading nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (214.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 214.1/214.1 MB 16.5 MB/s  0:00:12
Downloading nvidia_cufile-1.15.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 40.0 MB/s  0:00:00
Downloading nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl (59.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.5/59.5 MB 28.8 MB/s  0:00:02
Downloading nvidia_cusolver-12.0.4.66-py3-none-manylinux_2_27_x86_64.whl (200.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200.9/200.9 MB 16.4 MB/s  0:00:12
Downloading nvidia_cusparse-12.6.3.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (145.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 145.9/145.9 MB 18.2 MB/s  0:00:08
Downloading nvidia_nvjitlink-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (40.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.7/40.7 MB 36.0 MB/s  0:00:01
Downloading nvidia_nvtx-13.0.85-py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (148 kB)
Downloading setuptools-81.0.0-py3-none-any.whl (1.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 36.9 MB/s  0:00:00
Downloading torchvision-0.27.0-cp312-cp312-manylinux_2_28_x86_64.whl (7.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.6/7.6 MB 29.2 MB/s  0:00:00
Downloading torchaudio-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 43.4 MB/s  0:00:00
Downloading ollama-0.6.2-py3-none-any.whl (15 kB)
Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Downloading aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
Downloading attrs-26.1.0-py3-none-any.whl (67 kB)
Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)
Downloading certifi-2026.4.22-py3-none-any.whl (135 kB)
Downloading click-8.4.0-py3-none-any.whl (116 kB)
Downloading filelock-3.29.0-py3-none-any.whl (39 kB)
Downloading frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (242 kB)
Downloading fsspec-2026.4.0-py3-none-any.whl (203 kB)
Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading jsonpointer-3.1.1-py3-none-any.whl (7.7 kB)
Downloading langchain_protocol-0.0.15-py3-none-any.whl (7.0 kB)
Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (22 kB)
Using cached mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
Downloading networkx-3.6.1-py3-none-any.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 81.7 MB/s  0:00:00
Downloading nltk-3.9.4-py3-none-any.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 55.3 MB/s  0:00:00
Downloading orjson-3.11.9-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (134 kB)
Downloading ormsgpack-1.12.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)
Downloading packaging-26.2-py3-none-any.whl (100 kB)
Downloading propcache-0.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (61 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading python_dotenv-1.2.2-py3-none-any.whl (22 kB)
Downloading regex-2026.5.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (801 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 801.2/801.2 kB 19.6 MB/s  0:00:00
Using cached requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (507 kB)
Downloading scikit_learn-1.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (8.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.9/8.9 MB 68.0 MB/s  0:00:00
Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
Downloading scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (35.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.2/35.2 MB 17.9 MB/s  0:00:01
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading sympy-1.14.0-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 103.3 MB/s  0:00:00
Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 34.6 MB/s  0:00:00
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading tqdm-4.67.3-py3-none-any.whl (78 kB)
Downloading typer-0.25.1-py3-none-any.whl (58 kB)
Downloading annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
Downloading rich-15.0.0-py3-none-any.whl (310 kB)
Downloading pygments-2.20.0-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 77.4 MB/s  0:00:00
Downloading markdown_it_py-4.2.0-py3-none-any.whl (91 kB)
Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Downloading werkzeug-3.1.8-py3-none-any.whl (226 kB)
Downloading xxhash-3.7.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (193 kB)
Using cached zstandard-0.25.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (5.5 MB)
Downloading anyio-4.13.0-py3-none-any.whl (114 kB)
Downloading sniffio-1.3.1-py3-none-any.whl (10 kB)
Installing collected packages: torchaudio, nvidia-cusparselt-cu13, mpmath, cuda-toolkit, zstandard, xxhash, uuid-utils, urllib3, typing-inspection, triton, tqdm, threadpoolctl, tenacity, sympy, SQLAlchemy, sniffio, six, shellingham, setuptools, safetensors, regex, pyyaml, python-dotenv, python-bidi, pygments, pydantic-core, propcache, pillow, packaging, ormsgpack, orjson, nvidia-nvtx, nvidia-nvshmem-cu13, nvidia-nvjitlink, nvidia-nccl-cu13, nvidia-curand, nvidia-cufile, nvidia-cuda-runtime, nvidia-cuda-nvrtc, nvidia-cuda-cupti, numpy, networkx, mypy-extensions, multidict, mdurl, markupsafe, langchain-protocol, jsonpointer, joblib, itsdangerous, idna, httpx-sse, hf-xet, h11, fsspec, frozenlist, filelock, cuda-pathfinder, click, charset_normalizer, certifi, blinker, attrs, arabic-reshaper, annotated-types, annotated-doc, aiohappyeyeballs, yarl, werkzeug, typing-inspect, scipy, requests, python-dateutil, pydantic, nvidia-cusparse, nvidia-cufft, nvidia-cublas, nltk, marshmallow, markdown-it-py, jsonpatch, jinja2, httpcore, elastic-transport, cuda-bindings, anyio, aiosignal, scikit-learn, rich, requests-toolbelt, pydantic-settings, parsivar, pandas, nvidia-cusolver, nvidia-cudnn-cu13, httpx, flask, elasticsearch, dataclasses-json, aiohttp, typer, Ollama, langsmith, langgraph-sdk, flask-cors, torch, langchain-core, huggingface-hub, torchvision, tokenizers, langgraph-checkpoint, langchain-text-splitters, transformers, langgraph-prebuilt, langchain-classic, sentence-transformers, langgraph, langchain-community, langchain

Successfully installed Ollama-0.6.2 SQLAlchemy-2.0.49 aiohappyeyeballs-2.6.1 aiohttp-3.13.5 aiosignal-1.4.0 annotated-doc-0.0.4 annotated-types-0.7.0 anyio-4.13.0 arabic-reshaper-3.0.1 attrs-26.1.0 blinker-1.9.0 certifi-2026.4.22 charset_normalizer-3.4.7 click-8.4.0 cuda-bindings-13.2.0 cuda-pathfinder-1.5.4 cuda-toolkit-13.0.2 dataclasses-json-0.6.7 elastic-transport-9.4.0 elasticsearch-9.4.0 filelock-3.29.0 flask-3.1.3 flask-cors-6.0.2 frozenlist-1.8.0 fsspec-2026.4.0 h11-0.16.0 hf-xet-1.5.0 httpcore-1.0.9 httpx-0.28.1 httpx-sse-0.4.3 huggingface-hub-1.15.0 idna-3.15 itsdangerous-2.2.0 jinja2-3.1.6 joblib-1.5.3 jsonpatch-1.33 jsonpointer-3.1.1 langchain-1.3.1 langchain-classic-1.0.7 langchain-community-0.4.1 langchain-core-1.4.0 langchain-protocol-0.0.15 langchain-text-splitters-1.1.2 langgraph-1.2.0 langgraph-checkpoint-4.1.0 langgraph-prebuilt-1.1.0 langgraph-sdk-0.3.14 langsmith-0.8.5 markdown-it-py-4.2.0 markupsafe-3.0.3 marshmallow-3.26.2 mdurl-0.1.2 mpmath-1.3.0 multidict-6.7.1 mypy-extensions-1.1.0 networkx-3.6.1 nltk-3.9.4 numpy-2.4.5 nvidia-cublas-13.1.1.3 nvidia-cuda-cupti-13.0.85 nvidia-cuda-nvrtc-13.0.88 nvidia-cuda-runtime-13.0.96 nvidia-cudnn-cu13-9.20.0.48 nvidia-cufft-12.0.0.61 nvidia-cufile-1.15.1.6 nvidia-curand-10.4.0.35 nvidia-cusolver-12.0.4.66 nvidia-cusparse-12.6.3.3 nvidia-cusparselt-cu13-0.8.1 nvidia-nccl-cu13-2.29.7 nvidia-nvjitlink-13.0.88 nvidia-nvshmem-cu13-3.4.5 nvidia-nvtx-13.0.85 orjson-3.11.9 ormsgpack-1.12.2 packaging-26.2 pandas-3.0.3 parsivar-0.2.3.1 pillow-12.2.0 propcache-0.5.2 pydantic-2.13.4 pydantic-core-2.46.4 pydantic-settings-2.14.1 pygments-2.20.0 python-bidi-0.6.10 python-dateutil-2.9.0.post0 python-dotenv-1.2.2 pyyaml-6.0.3 regex-2026.5.9 requests-2.34.2 requests-toolbelt-1.0.0 rich-15.0.0 safetensors-0.7.0 scikit-learn-1.8.0 scipy-1.17.1 sentence-transformers-5.5.0 setuptools-81.0.0 shellingham-1.5.4 six-1.17.0 sniffio-1.3.1 sympy-1.14.0 tenacity-9.1.4 threadpoolctl-3.6.0 tokenizers-0.22.2 torch-2.12.0 torchaudio-2.11.0 torchvision-0.27.0 tqdm-4.67.3 transformers-5.8.1 triton-3.7.0 typer-0.25.1 typing-inspect-0.9.0 typing-inspection-0.4.2 urllib3-2.7.0 uuid-utils-0.15.0 werkzeug-3.1.8 xxhash-3.7.0 yarl-1.23.0 zstandard-0.25.0
3. Configure the engine in `src/search_engine/config.py` (ES host, model paths, etc.).
4. Start the server:

The UI will be accessible at `http://localhost:9002`.

### Running With Docker

You can use the provided Dockerfile and Docker Compose to run the system:

1. Ensure Docker and Docker Compose are installed.
2. Build and start the containers:


The UI will be accessible at `http://localhost:9002`.

## API Endpoints

- `POST /api/search`: Unified search for text and categories.
- `POST /api/process-image`: Multimodal search using image uploads.
- `POST /api/process-query`: Query cleaning, keyboard fix, and spell correction.
- `POST /api/detect-categories`: Suggest relevant categories for a query.
- `GET /api/reload-user/<user_id>`: Refresh user profile from CSV.

## Project Structure

- `src/search_engine/`: Main package.
  - `nlp/`: Persian NLP, spelling, keyboard fix.
  - `llm/`: Query enhancement, catalog context.
  - `retrieval/`: Sparse, dense, and hybrid retrieval.
  - `reranking/`: RRF and cross-encoder logic.
  - `personalization/`: User history and boosts.
  - `multimodal/`: Image captioning.
  - `api/`: Flask app and route handlers.
- `run_server.py`: Main entry point script.
- `user_history.csv`: User history data for personalization.
