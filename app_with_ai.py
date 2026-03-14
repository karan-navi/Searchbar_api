import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 1. SETUP CONNECTIONS
ES_URL = "https://d42c203b6e464b8ab4102ff12307a4d1.us-central1.gcp.cloud.es.io:443"
elastic_password = os.getenv("ELASTIC_PASSWORD")
es = Elasticsearch(ES_URL, basic_auth=("elastic", elastic_password))

# Load the AI model once at startup
print("Loading AI Model for Hybrid Search...")
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return "SEARCH API IS ONLINE. To search anything open with http://127.0.0.1:5000/search?q={Anything you want to search}"


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    
    # --- STEP 1: PAGINATION LOGIC ---
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('pageSize', 10))
        if page < 1: page = 1
        if page_size < 1: page_size = 10
        if page_size > 100: page_size = 100
    except ValueError:
        page = 1
        page_size = 10

    es_from = (page - 1) * page_size

    if not query:
        return jsonify({"success": False, "message": "No query provided"}), 400

    try:
        # --- STEP 2: GENERATE VECTOR ---
        query_vector = model.encode(query).tolist()

        # --- STEP 3: EXECUTE HYBRID SEARCH ---
        # This combines Keyword (multi_match) + Vector (knn)
        response = es.search(
            index="shops_index",
            from_=es_from,
            size=page_size,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["shopName^3", "category^2", "services", "vendor_name"],
                    "fuzziness": "AUTO"
                }
            },
            knn={
                "field": "shop_vector",
                "query_vector": query_vector,
                "k": 10,
                "num_candidates": 100,
                "boost": 0.8  # Balance between text and vector
            }
        )

        # --- STEP 4: FORMAT RESULTS ---
        results = [hit["_source"] for hit in response["hits"]["hits"]]
        total_count = response["hits"]["total"]["value"]

        return jsonify({
            "success": True,
            "code": "success",
            "message": "Shops retrieved successfully." if total_count > 0 else f"No results for '{query}'",
            "data": {
                "items": results,
                "totalCount": total_count,
                "page": page,
                "pageSize": page_size
            }
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)