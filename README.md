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
2. Install dependencies
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
