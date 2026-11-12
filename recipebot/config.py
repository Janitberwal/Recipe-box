from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "recipes.db"
SOURCES = {
    "ar": "recipes_raw_nosource_ar.json",   # Allrecipes
    "epi": "recipes_raw_nosource_epi.json",  # Epicurious
    "fn": "recipes_raw_nosource_fn.json",   # Food Network
}
SOURCE_NAMES = {"ar": "Allrecipes", "epi": "Epicurious", "fn": "Food Network"}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
# used automatically if the primary model is overloaded, retired or times out
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash").split(",") if m.strip()]
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low").strip()   # low = fast intent parsing
