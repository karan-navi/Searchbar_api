import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import chromadb

app = Flask(__name__)
CORS(app)

print("Loading AI Model for Search...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to local ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./shop_vector_db")
collection = chroma_client.get_collection(name="shops_index")

@app.route('/')
def home():
    return "SEARCH API IS ONLINE. To search anything open with http://127.0.0.1:5000/search?q={query}"

@app.route('/search', methods=['GET'])
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

    if not query:
        return jsonify({"success": False, "message": "No query provided"}), 400

    try:
        query_vector = model.encode(query).tolist()

        max_candidates = 50
        
        response = collection.query(
            query_embeddings=[query_vector],
            n_results=max_candidates
        )
        items = []
        total_valid_matches = 0

        MAX_DISTANCE_THRESHOLD = 0.65 

        if response['ids'] and len(response['ids']) > 0:
            ids = response['ids'][0]
            metadatas = response['metadatas'][0]
            distances = response['distances'][0] 
            
            valid_results = []

            for i in range(len(ids)):
                # Only keep results that are mathematically similar to the query
                if distances[i] <= MAX_DISTANCE_THRESHOLD:
                    raw_meta = metadatas[i]
                    entity_type = raw_meta.get("entity_type")
                    match_confidence = round((1 - distances[i]) * 100, 1)
                    
                    # --- FORMAT A SHOP ---
                    if entity_type == "Shop":
                        formatted_item = {
                            "entity_type": "Shop",
                            "shopId": raw_meta.get("shopId"),
                            "shopName": raw_meta.get("shopName"),
                            "category": raw_meta.get("category"),
                            "address": raw_meta.get("address"),
                            "phones": raw_meta.get("phones"),
                            "whatsapp": raw_meta.get("whatsapp"),
                            "images": raw_meta.get("images"),
                            "services": raw_meta.get("services").split(', ') if raw_meta.get("services") else [],
                            "rating": raw_meta.get("rating"),
                            "review_count": raw_meta.get("review_count"),
                            "is_verified": raw_meta.get("is_verified"),
                            "match_confidence_percent": match_confidence
                        }
                    
                    # --- FORMAT A VENDOR ---
                    elif entity_type == "Vendor":
                        formatted_item = {
                            "entity_type": "Vendor",
                            "vendorId": raw_meta.get("Id"),
                            "vendorName": raw_meta.get("Name"),
                            "occupation_note": raw_meta.get("Note"),
                            "experience_years": raw_meta.get("YearOfExperience"),
                            "mobileNo": raw_meta.get("MobileNo"),
                            "images": raw_meta.get("VendorImages"),
                            "rating": raw_meta.get("rating"),
                            "review_count": raw_meta.get("review_count"),
                            "match_confidence_percent": match_confidence
                        }
                    else:
                        continue
                    
                    valid_results.append(formatted_item)

            total_valid_matches = len(valid_results)
            
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            items = valid_results[start_idx:end_idx]

        return jsonify({
            "success": True,
            "code": "success",
            "message": "Shops retrieved successfully." if items else f"No relevant results found for '{query}'.",
            "data": {
                "items": items,
                "totalCount": total_valid_matches,
                "page": page,
                "pageSize": page_size
            }
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
