"""
model_loader.py
---------------
Loads trained ML models and metadata once at startup.
All routes import from here so models are not reloaded per request.
"""

import os
import pickle
import json

# ── Paths — models folder is now INSIDE backend/ ──
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BACKEND_DIR, 'models')
MODELS_PKL  = os.path.join(MODELS_DIR, 'trained_models.pkl')
META_JSON   = os.path.join(MODELS_DIR, 'model_meta.json')

# ── All 8 valid food keys ──────────────────────────
VALID_FOODS = {
    'rice'  : 'rice_NGN',
    'beans' : 'beans_NGN',
    'garri' : 'garri_NGN',
    'yam'   : 'yam_NGN',
    'eggs'  : 'eggs',
    'fish'  : 'fish',
    'beef'  : 'meat_beef',
    'milk'  : 'milk',
}

print("[ModelLoader] Loading trained models from:", MODELS_DIR)
with open(MODELS_PKL, 'rb') as f:
    MODELS = pickle.load(f)

with open(META_JSON, 'r') as f:
    META = json.load(f)

print(f"[ModelLoader] Loaded models for: {list(MODELS.keys())}")


def get_model(food_key: str):
    return MODELS.get(food_key)


def get_meta(food_key: str = None):
    if food_key:
        return META.get(food_key)
    return META


def resolve_food(food_name: str):
    key      = food_name.lower().strip()
    internal = VALID_FOODS.get(key)
    if not internal:
        return None, f"Unknown food '{food_name}'. Valid options: {list(VALID_FOODS.keys())}"
    return internal, None
