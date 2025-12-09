import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import MONGO_URI, DB_NAME
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Test connection
try:
    collections = db.list_collection_names()
    print("Connected to MongoDB successfully!")
    print("Collections:", collections)
except Exception as e:
    print("Connection failed:", e)
