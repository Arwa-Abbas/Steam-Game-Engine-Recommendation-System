import sys
import os
from pymongo import MongoClient

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import MONGO_URI, DB_NAME
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    print(f"✅ Connected to MongoDB database: {DB_NAME}")
    print(f"📊 Available collections: {db.list_collection_names()}")
    
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {str(e)}")
    raise