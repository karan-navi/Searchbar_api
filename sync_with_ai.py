import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import urllib.parse

load_dotenv()

# --- 1. CONFIGURATION ---
SERVER = '69.62.78.150,1433' 
DATABASE = 'eazypeazzy-dev'
USERNAME='dev'
PASSWORD= os.getenv('PASSWORD')

if PASSWORD is None:
    # This will help you debug if the .env isn't loading
    print("Error: SQL_PASSWORD not found in .env file!")

safe_password = urllib.parse.quote_plus(PASSWORD)


INDEX_NAME = 'shops_index'
ES_URL = "https://d42c203b6e464b8ab4102ff12307a4d1.us-central1.gcp.cloud.es.io:443"

elastic_password = os.getenv("ELASTIC_PASSWORD")
es = Elasticsearch(ES_URL, basic_auth=("elastic", elastic_password))

db_uri = f"mssql+pyodbc://{USERNAME}:{safe_password}@{SERVER}/{DATABASE}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
engine = create_engine(db_uri)

INDEX_NAME = "shops_index"

print("Loading AI Model (this might take a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_index():
    """Deletes the old index and creates a new one with strict Vector mapping."""
    if es.indices.exists(index=INDEX_NAME):
        print(f"Deleting old index '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)

    print("Creating new index with vector mapping...")
    mapping = {
        "mappings": {
            "properties": {
                "shop_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine" # Measures the angle between vectors to find similarities
                },
                "shopName": {"type": "text"},
                "shopCategory": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)

def sync_data():
    """Pulls data from SQL, generates AI vectors, and pushes to Elasticsearch."""
    print("Fetching data from SQL Server...")
    
    query = text("""
        SELECT 
            s.ShopId, s.ShopName, s.ShopCategory, s.ShopAddress, s.ShopPhones, 
            s.ShopWhatsappNo, s.ShopImages, s.IsActive, s.IsVerified,
            sc.ShopCategoryName,
            v.Name as VendorName, v.YearOfExperience,
            STRING_AGG(CAST(ser.Name AS VARCHAR(MAX)), ', ') as ServiceList,
            AVG(CAST(vr.Rating AS FLOAT)) as AvgRating,
            COUNT(vr.ReviewID) as TotalReviews
        FROM dbo.Shops s
        LEFT JOIN dbo.shopCategories sc ON s.ShopCategory = sc.ShopCategoryName
        LEFT JOIN dbo.Vendors v ON s.ShopId = v.Id 
        LEFT JOIN dbo.Services ser ON s.ShopCategory = ser.Category 
        LEFT JOIN dbo.VendorReviews vr ON v.Id = vr.VendorID -- Fixed spelling here (o instead of e)
        GROUP BY 
            s.ShopId, s.ShopName, s.ShopCategory, s.ShopAddress, s.ShopPhones, 
            s.ShopWhatsappNo, s.ShopImages, s.IsActive, s.IsVerified,
            sc.ShopCategoryName, v.Name, v.YearOfExperience
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    if not rows:
        print("No data found in SQL Server!")
        return

    print(f"Found {len(rows)} shops. Generating AI vectors and preparing upload...")
    actions = []

    for row in rows:
        shop_name = row.ShopName if row.ShopName else "Unknown Shop"
        category = row.ShopCategoryName if row.ShopCategoryName else (row.ShopCategory if row.ShopCategory else "")
        vendor = row.VendorName if row.VendorName else ""
        services = row.ServiceList if row.ServiceList else ""
        address = row.ShopAddress if row.ShopAddress else ""

        text_to_vectorize = f"Shop: {shop_name}. Category: {category}. Vendor: {vendor}. Services: {services}. Location: {address}"

        vector = model.encode(text_to_vectorize).tolist()

        shop_doc = {
            "shopId": str(row.ShopId),
            "shopName": shop_name,
            "category": category,
            "address": address,
            "phones": row.ShopPhones,
            "whatsapp": row.ShopWhatsappNo,
            "images": row.ShopImages,
            "vendor_name": vendor,
            "experience": row.YearOfExperience,
            "services": services.split(', ') if services else [],
            "rating": round(float(row.AvgRating or 0), 1),
            "review_count": int(row.TotalReviews or 0),
            "is_verified": bool(row.IsVerified),
            "shop_vector": vector 
        }

        actions.append({
            "_index": INDEX_NAME,
            "_id": shop_doc["shopId"],
            "_source": shop_doc
        })

    print("Uploading to Elasticsearch...")
    helpers.bulk(es, actions)
    print("Sync complete! Your data is now AI-ready.")

if __name__ == "__main__":
    create_index()
    sync_data()