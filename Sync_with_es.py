import pyodbc
from elasticsearch import Elasticsearch, helpers

# --- Configurations ---
SERVER = '69.62.78.150,1433' 
DATABASE = 'eazypeazzy-dev'
INDEX_NAME = 'shops_index'
ES_URL = "https://d42c203b6e464b8ab4102ff12307a4d1.us-central1.gcp.cloud.es.io:443"

conn_str = (
    r'DRIVER={ODBC Driver 18 for SQL Server};'
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    r'UID=dev;'
    r'PWD=Eazyp!0225@Dv;'
    r'TrustServerCertificate=yes;'
)

es = Elasticsearch(
    ES_URL,
    basic_auth=("elastic", "9sv4oTd8kFgsQ7nsdhf8njaS")
)

# 1. Quick Connection Test
try:
    if es.ping():
        print("Connected to Elastic Cloud successfully!")
    else:
        print("Connection failed. Check your password.")
        exit()
except Exception as e:
    print(f"Could not connect: {e}")
    exit()

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

def fetch_and_format_data():
    cursor.execute("SELECT * FROM vw_Elasticsearch_Shops")
    columns = [column[0] for column in cursor.description]
    
    for row in cursor.fetchall():
        doc = dict(zip(columns, row))
        
        created_dt = doc.get("createdDate")
        modified_dt = doc.get("modifiedDate")
        created_str = created_dt.isoformat() if created_dt else ""
        modified_str = modified_dt.isoformat() if modified_dt else ""

        yield {
            "_index": INDEX_NAME,
            "_id": str(doc["shopId"]),
            "_source": {
                "shopId": str(doc["shopId"]),
                "shopCategory": str(doc.get("shopCategory", "")),
                "shopName": str(doc.get("shopName", "")),
                "shopAddress": str(doc.get("shopAddress", "")),
                "shopPhones": str(doc.get("shopPhones", "")),
                "shopWhatsappNo": str(doc.get("shopWhatsappNo", "")),
                "shopImages": str(doc.get("shopImages", "")),
                "isActive": bool(doc.get("isActive", False)),
                "isVerified": bool(doc.get("isVerified", False)),
                "createdDate": created_str,
                "modifiedDate": modified_str
            }
        }

try:
    print("Starting sync to Cloud...")
    success, failed = helpers.bulk(es, fetch_and_format_data())
    print(f"Successfully synced {success} documents.")
except Exception as e:
    print(f"Error during sync: {e}")
finally:
    cursor.close()
    conn.close()