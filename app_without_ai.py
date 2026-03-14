import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
ES_URL = "https://d42c203b6e464b8ab4102ff12307a4d1.us-central1.gcp.cloud.es.io:443"

elastic_password = os.getenv("ELASTIC_PASSWORD")

# print(elastic_password)

es = Elasticsearch(ES_URL, basic_auth=("elastic", elastic_password))
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return "SERVER IS ALIVE! Try going to /search?q=gaurav"

@app.route('/search')
def search():
    query = request.args.get('q', '')
    
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

    # Check for empty query
    if not query:
        return jsonify({
            "success": False,
            "code": "missing_query",
            "message": "Please provide a search term.",
            "data": {"items": [], "totalCount": 0, "page": page, "pageSize": page_size}
        }), 400

    try:
        # Convert the user's text into an AI vector
        query_vector = model.encode(query).tolist()

        response = es.search(
            index="shops_index", 
            from_=es_from,
            size=page_size,
            knn={
                "field": "shop_vector", 
                "query_vector": query_vector,
                "k": page_size,  
                "num_candidates": 100
            }
        )
        
        results = [hit["_source"] for hit in response["hits"]["hits"]]
        total_count = response["hits"]["total"]["value"]

        if total_count == 0:
            custom_message = f"No shops found matching '{query}'."
        else:
            custom_message = "Shops retrieved successfully."

        return jsonify({
            "success": True,
            "code": "success",
            "message": custom_message,
            "data": {
                "items": results,
                "totalCount": total_count,
                "page": page,
                "pageSize": page_size
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "code": "server_error",
            "message": str(e),
            "data": None
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)