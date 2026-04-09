import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import urllib.parse
import chromadb

load_dotenv()

# --- 1. CONFIGURATION ---
SERVER = '69.62.78.150,1433' 
DATABASE = 'eazypeazzy-dev'
USERNAME = 'dev'
PASSWORD = os.getenv('PASSWORD').strip()

odbc_str = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={{{PASSWORD}}};TrustServerCertificate=yes"
encoded_odbc = urllib.parse.quote_plus(odbc_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_odbc}")

print("Loading AI Model (this might take a few seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./shop_vector_db")

def setup_collection():
    try:
        chroma_client.delete_collection(name="shops_index")
        print("Deleted old local index.")
    except Exception:
        pass 
    print("Creating new local vector collection...")
    return chroma_client.create_collection(
        name="shops_index",
        metadata={"hnsw:space": "cosine"}
    )

def sync_data():
    collection = setup_collection()
    ids, embeddings, metadatas = [], [], []

    with engine.connect() as conn:
        # ==========================================
        # 1. FETCH ALL INDEPENDENT SHOPS
        # ==========================================
        print("Fetching SHOPS from SQL Server...")
        shop_query = text("""
            SELECT 
                s.ShopId, s.ShopName, s.ShopCategory, s.ShopAddress, s.ShopPhones, 
                s.ShopWhatsappNo, s.ShopImages, s.IsActive, s.IsVerified,
                STRING_AGG(CAST(ser.Name AS VARCHAR(MAX)), ', ') as ServiceList
            FROM dbo.Shops s
            LEFT JOIN dbo.Services ser ON s.ShopCategory = ser.Category 
            GROUP BY 
                s.ShopId, s.ShopName, s.ShopCategory, s.ShopAddress, s.ShopPhones, 
                s.ShopWhatsappNo, s.ShopImages, s.IsActive, s.IsVerified
        """)
        shops = conn.execute(shop_query).fetchall()

        for row in shops:
            name = row.ShopName or "Unknown Shop"
            cat = row.ShopCategory or ""
            services = row.ServiceList or ""
            
            text_to_vectorize = f"Entity: Shop. Name: {name}. Category: {cat}. Services: {services}."
            vector = model.encode(text_to_vectorize).tolist()

            ids.append(f"SHOP_{row.ShopId}") # Prefix to avoid ID collisions
            embeddings.append(vector)
            metadatas.append({
                "entity_type": "Shop",
                "shopId": str(row.ShopId),
                "shopName": name,
                "category": cat,
                "address": row.ShopAddress or "",
                "phones": str(row.ShopPhones or ""),
                "whatsapp": str(row.ShopWhatsappNo or ""),
                "images": str(row.ShopImages or ""),
                "vendor_name": "",
                "vendor_note": "",
                "experience": 0,
                "services": services,
                "rating": 0.0,
                "review_count": 0,
                "is_verified": bool(row.IsVerified)
            })

        # ==========================================
        # 2. FETCH ALL INDEPENDENT VENDORS
        # ==========================================
        print("Fetching VENDORS from SQL Server...")
        vendor_query = text("""
            SELECT 
                v.Id, v.Name, v.Note, v.MobileNo, v.YearOfExperience, v.isActive,
                AVG(CAST(vr.Rating AS FLOAT)) as AvgRating,
                COUNT(vr.ReviewID) as TotalReviews
            FROM dbo.Vendors v
            LEFT JOIN dbo.VendorReviews vr ON v.Id = vr.VendorID
            GROUP BY 
                v.Id, v.Name, v.Note, v.MobileNo, v.YearOfExperience, v.isActive
        """)
        vendors = conn.execute(vendor_query).fetchall()

        for row in vendors:
            name = row.Name or "Unknown Vendor"
            note = row.Note or ""
            
            # The AI explicitly reads the Note to know their occupation!
            text_to_vectorize = f"Entity: Independent Professional. Name: {name}. Occupation/Notes: {note}."
            vector = model.encode(text_to_vectorize).tolist()

            ids.append(f"VENDOR_{row.Id}")
            embeddings.append(vector)
            
            # We map vendor Name to 'shopName' so your frontend UI still displays the title correctly
            metadatas.append({
                "entity_type": "Vendor",
                "shopId": str(row.Id),
                "shopName": name, 
                "category": note, 
                "address": "",
                "phones": str(row.MobileNo or ""),
                "whatsapp": "",
                "images": "",
                "vendor_name": name,
                "vendor_note": note,
                "experience": int(row.YearOfExperience or 0),
                "services": "",
                "rating": float(round(row.AvgRating or 0, 1)),
                "review_count": int(row.TotalReviews or 0),
                "is_verified": False
            })

    total_records = len(ids)
    print(f"Total entities found: {len(shops)} Shops + {len(vendors)} Vendors = {total_records} Total")

    if total_records == 0:
        print("No data found to sync.")
        return

    print("Uploading to local ChromaDB...")
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print("✅ Sync complete! Your independent Shops and Vendors are now AI-ready.")

if __name__ == "__main__":
    sync_data()
