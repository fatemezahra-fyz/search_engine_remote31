# src/search_engine/api/app.py

import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.search_engine.core import SearchEngine
from src.search_engine.config import INDEX_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
CORS(app)
search_engine = None

def get_engine():
    global search_engine
    if search_engine is None:
        logger.info("Initializing SearchEngine in API server...")
        search_engine = SearchEngine()
        search_engine.personalization.load_history()
    return search_engine

@app.route('/')
def index():
    logger.info("Serving UI index page")
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/process-query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query', '')
    if not query:
        logger.warning("Query processing requested without query string")
        return jsonify({'error': 'Query is required'}), 400

    logger.info(f"API: Processing query: '{query}'")
    result = get_engine().process_query_complete(query)
    return jsonify(result)

@app.route('/api/detect-categories', methods=['POST'])
def detect_categories():
    data = request.json
    query = data.get('query', '')
    if not query:
        logger.warning("Category detection requested without query string")
        return jsonify({'error': 'Query is required'}), 400

    logger.info(f"API: Detecting categories for: '{query}'")
    cat_scores = get_engine().category_detector.detect(query)
    categories = [{'name': c[0], 'combined': float(c[1]), 'count': int(c[2]), 'similarity': float(c[3])} for c in cat_scores]
    return jsonify({'categories': categories})

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    category = data.get('category')
    user_id = data.get('user_id')

    if not query:
        logger.warning("Search requested without query string")
        return jsonify({'error': 'Query is required'}), 400

    logger.info(f"API: Search request - query: '{query}', category: '{category}', user: '{user_id}'")
    engine = get_engine()
    catalog_context = engine.catalog_provider.build_catalog_context(category)
    results = engine.unified_search(query, category, user_id, catalog_context=catalog_context)

    formatted = []
    for hit in results:
        src = hit.get('_source', {})
        formatted.append({
            'id': hit.get('_id'),
            'title': src.get('product_title_fa', 'N/A'),
            'brand': src.get('brand_name_fa', 'N/A'),
            'shop': src.get('shop', 'N/A'),
            'category': src.get('category_keywords', 'N/A'),
            'price': str(src.get('price', 'N/A')),
            'score': float(hit.get('final_score', hit.get('_score', 0.0))),
            'url_code': src.get('url_code', '#')
        })
    logger.info(f"API: Search returned {len(formatted)} results")
    return jsonify({'results': formatted, 'total': len(formatted)})

@app.route('/api/reload-user/<user_id>', methods=['POST', 'GET'])
def reload_user(user_id):
    logger.info(f"API: Reloading profile for user: {user_id}")
    success = get_engine().personalization.load_user_profile(user_id)
    if success:
        logger.info(f"API: Successfully reloaded user {user_id}")
        return jsonify({'status': 'success', 'message': f'Profile for user {user_id} reloaded'})
    logger.warning(f"API: Failed to reload user {user_id}")
    return jsonify({'status': 'error', 'message': f'User {user_id} not found'}), 404

@app.route('/api/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        logger.warning("Image processing requested without image file")
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)
    logger.info(f"API: Processing uploaded image: {filename}")

    try:
        engine = get_engine()
        caption = engine.image_captioner.generate_caption(filepath)
        logger.info(f"API: Generated image caption: {caption}")
        persian_query = engine.caption_enhancer.enhance_caption(caption)
        logger.info(f"API: Enhanced Persian query: {persian_query}")
        return jsonify({'english_caption': caption, 'persian_query': persian_query})
    except Exception as e:
        logger.error(f"API: Image processing failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def run():
    logger.info("🚀 Starting Flask API server on port 9002")
    app.run(host='0.0.0.0', port=9002)

if __name__ == '__main__':
    run()
