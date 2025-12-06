from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from bson import json_util
from contextlib import asynccontextmanager

from config import PORT
from src.db import db
from src.recommender import GameRecommender

# Initialize recommender globally
recommender = GameRecommender(db)

# Pydantic models
class SystemSpecs(BaseModel):
    memory_gb: Optional[int] = None
    storage_gb: Optional[int] = None
    os_type: Optional[str] = None

class UserPreferences(BaseModel):
    max_price: float = 50.0
    preferred_tags: List[str] = []
    languages: List[str] = []
    developers: List[str] = []
    publishers: List[str] = []
    system_specs: Optional[SystemSpecs] = None

class RecommendationRequest(BaseModel):
    preferences: UserPreferences
    limit: int = 10

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown"""
    print("🚀 Starting Game Recommendation API...")
    print("📊 Loading recommendation model...")
    
    # Load pre-trained model or train new one
    recommender.load_model()
    
    print("✅ Recommendation system ready!")
    
    yield
    
    print("👋 Shutting down...")

app = FastAPI(
    title="Game Recommendation API",
    description="Knowledge-based game recommendation system",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_mongo_data(data):
    """Convert MongoDB documents to JSON"""
    return json.loads(json_util.dumps(data))

@app.get("/")
def root():
    return {
        "message": "Game Recommendation API v3.0",
        "status": "running",
        "endpoints": {
            "GET /health": "Health check",
            "GET /games": "Browse all games",
            "GET /games/search": "Search by title",
            "POST /games/filter": "Filter by criteria",
            "POST /recommend": "Get personalized recommendations",
            "GET /similar/{title}": "Find similar games",
            "GET /stats": "Database statistics",
            "GET /tags": "All available tags",
            "GET /languages": "All supported languages",
            "GET /developers": "All developers",
            "POST /retrain": "Retrain recommendation model"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "games_loaded": len(recommender.games),
        "model_trained": recommender.similarity_matrix is not None
    }

@app.get("/games")
def get_games(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("popularity_score", enum=[
        "popularity_score", "overall_sentiment_score", 
        "original_price", "all_reviews_count", "title"
    ]),
    sort_order: int = Query(-1, enum=[1, -1])
):
    """Get paginated games"""
    skip = (page - 1) * limit
    
    games = list(db.steam_games.find(
        {},
        {"_id": 0}
    ).sort(sort_by, sort_order).skip(skip).limit(limit))
    
    total = db.steam_games.count_documents({})
    
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": (total + limit - 1) // limit,
        "games": convert_mongo_data(games)
    }

@app.get("/games/search")
def search_games(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=50)
):
    """Search games by title"""
    games = list(db.steam_games.find(
        {"title_lower": {"$regex": q.lower(), "$options": "i"}},
        {"_id": 0}
    ).limit(limit))
    
    return {
        "query": q,
        "count": len(games),
        "games": convert_mongo_data(games)
    }

@app.post("/games/filter")
def filter_games(
    tags: List[str] = [],
    min_price: float = 0,
    max_price: float = 1000,
    languages: List[str] = [],
    categories: List[str] = [],
    limit: int = 20
):
    """Filter games by criteria"""
    query = {}
    
    # Price
    query["discounted_price"] = {"$gte": min_price, "$lte": max_price}
    
    # Tags
    if tags:
        query["tags"] = {"$in": [tag.lower() for tag in tags]}
    
    # Languages
    if languages:
        query["languages"] = {"$in": [lang.lower() for lang in languages]}
    
    # Categories
    if categories:
        query["categories"] = {"$in": [cat.lower() for cat in categories]}
    
    games = list(db.steam_games.find(
        query,
        {"_id": 0}
    ).sort("popularity_score", -1).limit(limit))
    
    return {
        "filters": {
            "tags": tags,
            "price_range": [min_price, max_price],
            "languages": languages,
            "categories": categories
        },
        "count": len(games),
        "games": convert_mongo_data(games)
    }

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    """Get personalized game recommendations"""
    try:
        # Convert preferences to dict
        prefs_dict = request.preferences.dict()
        
        # Convert SystemSpecs to dict if present
        if prefs_dict.get('system_specs'):
            prefs_dict['system_specs'] = prefs_dict['system_specs']
        
        print(f"🎯 Preferences received: {prefs_dict}")
        
        recommendations = recommender.recommend_by_preferences(
            prefs_dict,
            request.limit
        )
        
        if not recommendations:
            return {
                "recommendations": [],
                "count": 0,
                "message": "No games match your criteria. Try adjusting your preferences."
            }
        
        # Format recommendations
        formatted = []
        for rec in recommendations:
            game = rec['game']
            explanations = recommender.get_explanation(game, rec['score_breakdown'])
            
            formatted.append({
                "title": game.get("title"),
                "price": game.get("discounted_price", game.get("original_price", 0)),
                "original_price": game.get("original_price", 0),
                "discount": game.get("discount_percentage", 0),
                "score": round(rec['score'], 3),
                "sentiment": game.get("overall_sentiment_score", 0.5),
                "popularity": game.get("popularity_score", 0.3),
                "reviews_count": game.get("all_reviews_count", 0),
                "tags": game.get("tags", [])[:5],
                "categories": game.get("categories", []),
                "features": game.get("features", [])[:3],
                "languages": game.get("languages", [])[:5],
                "developer": game.get("developer"),
                "publisher": game.get("publisher"),
                "release_year": game.get("release_year"),
                "link": game.get("link"),
                "explanations": explanations,
                "score_breakdown": rec['score_breakdown'],
                "memory_gb": game.get("memory_gb"),
                "storage_gb": game.get("storage_gb"),
                "os_type": game.get("os_type")
            })
        
        return {
            "recommendations": formatted,
            "count": len(formatted),
            "preferences_used": prefs_dict
        }
        
    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/similar/{game_title}")
def get_similar_games(
    game_title: str,
    limit: int = Query(5, ge=1, le=20)
):
    """Find similar games"""
    try:
        recommendations = recommender.recommend_similar_games(game_title, limit)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"Game '{game_title}' not found")
        
        formatted = []
        for rec in recommendations:
            game = rec['game']
            formatted.append({
                "title": game.get("title"),
                "similarity": round(rec['similarity_score'], 3),
                "price": game.get("discounted_price", game.get("original_price", 0)),
                "sentiment": game.get("overall_sentiment_score", 0.5),
                "tags": game.get("tags", [])[:5],
                "developer": game.get("developer")
            })
        
        return {
            "source_game": game_title,
            "recommendations": formatted,
            "count": len(formatted)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Database statistics"""
    total = db.steam_games.count_documents({})
    
    # Price stats
    price_pipeline = [
        {"$group": {
            "_id": None,
            "avg_price": {"$avg": "$discounted_price"},
            "max_price": {"$max": "$discounted_price"},
            "min_price": {"$min": "$discounted_price"},
            "free_games": {"$sum": {"$cond": [{"$eq": ["$discounted_price", 0]}, 1, 0]}}
        }}
    ]
    price_stats = list(db.steam_games.aggregate(price_pipeline))
    
    # Top tags
    tag_pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 20}
    ]
    top_tags = list(db.steam_games.aggregate(tag_pipeline))
    
    # Top languages
    lang_pipeline = [
        {"$unwind": "$languages"},
        {"$group": {"_id": "$languages", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_languages = list(db.steam_games.aggregate(lang_pipeline))
    
    return {
        "total_games": total,
        "price_stats": price_stats[0] if price_stats else {},
        "top_tags": top_tags,
        "top_languages": top_languages
    }

@app.get("/tags")
def get_tags():
    """Get all unique tags"""
    tags = db.steam_games.distinct("tags")
    return {"tags": sorted(tags), "count": len(tags)}

@app.get("/languages")
def get_languages():
    """Get all supported languages"""
    languages = db.steam_games.distinct("languages")
    return {"languages": sorted(languages), "count": len(languages)}

@app.get("/developers")
def get_developers():
    """Get all developers"""
    developers = db.steam_games.distinct("developer")
    return {"developers": sorted([d for d in developers if d]), "count": len(developers)}

@app.get("/publishers")
def get_publishers():
    """Get all publishers"""
    publishers = db.steam_games.distinct("publisher")
    return {"publishers": sorted([p for p in publishers if p]), "count": len(publishers)}

@app.post("/retrain")
def retrain_model():
    """Retrain the recommendation model"""
    try:
        recommender.train_and_save_model()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "games_count": len(recommender.games)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)