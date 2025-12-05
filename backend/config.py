from dotenv import load_dotenv
import os
from pathlib import Path

# Path to project root (where .env is)
BASE_DIR = Path(__file__).resolve().parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
PORT = int(os.getenv("PORT", 8000))
