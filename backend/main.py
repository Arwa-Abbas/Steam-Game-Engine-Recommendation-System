from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from bson import json_util
from contextlib import asynccontextmanager

# Import your modules
from config import PORT
from src.db import db
from src.recommender import GameRecommender

# Initialize recommender globally
recommender = GameRecommender(db)

# Pydantic models
class UserPreferences(BaseModel):
    max_price: float = 100.0
    preferred_tags: List[str] = []
    preferred_categories: List[str] = []
    min_sentiment: float = 0.6
    min_popularity: float = 0.3
    system_specs: Optional[Dict[str, Any]] = {}

class RecommendationRequest(BaseModel):
    preferences: UserPreferences
    limit: int = 10

class FilterRequest(BaseModel):
    tags: Optional[List[str]] = []
    min_price: float = 0
    max_price: float = 1000
    min_sentiment: float = 0.0
    min_popularity: float = 0.0
    categories: Optional[List[str]] = []
    limit: int = 20

# Lifespan handler to replace deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup, cleanup on shutdown"""
    print("🚀 Starting Game Recommendation API...")
    print("📊 Initializing recommendation system...")
    
    # Load and prepare games
    recommender.load_games()
    recommender.prepare_features()
    
    print("✅ Recommendation system ready!")
    
    yield  # App runs here
    
    # Cleanup on shutdown (optional)
    print("👋 Shutting down recommendation system...")

app = FastAPI(
    title="Game Recommendation API",
    description="Knowledge-based game recommendation system",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to convert MongoDB data
def convert_mongo_data(data):
    """Convert MongoDB documents to JSON serializable format"""
    return json.loads(json_util.dumps(data))

@app.get("/")
def root():
    return {
        "message": "Game Recommendation API v2.0",
        "endpoints": {
            "GET /health": "Check API health",
            "GET /games": "Browse games with pagination",
            "GET /games/search": "Search games by title",
            "GET /games/filter": "Filter games by criteria",
            "POST /recommend/preferences": "Get recommendations based on preferences",
            "GET /recommend/similar/{game_title}": "Get similar games",
            "GET /stats": "Get database statistics",
            "GET /tags": "Get all unique tags",
            "GET /categories": "Get all unique categories"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "games_loaded": len(recommender.games)
    }

@app.get("/games")
def get_games(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("popularity_score", enum=[
        "popularity_score", "overall_sentiment_score", 
        "original_price", "all_reviews_count", "release_year"
    ]),
    sort_order: int = Query(-1, enum=[1, -1])
):
    """Get paginated games with sorting"""
    skip = (page - 1) * limit
    
    # Build sort field
    sort_field = sort_by
    if sort_by == "release_year" and sort_order == -1:
        sort_field = "release_year"
    
    games = list(db.steam_games.find(
        {},
        {
            "_id": 0,
            "title": 1,
            "original_price": 1,
            "overall_sentiment_score": 1,
            "popularity_score": 1,
            "tags": 1,
            "developer": 1,
            "release_year": 1,
            "categories": 1,
            "all_reviews_count": 1
        }
    ).sort(sort_field, sort_order).skip(skip).limit(limit))
    
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
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=50)
):
    """Search games by title (case-insensitive)"""
    games = list(db.steam_games.find(
        {"title_lower": {"$regex": q.lower(), "$options": "i"}},
        {
            "_id": 0,
            "title": 1,
            "original_price": 1,
            "overall_sentiment_score": 1,
            "popularity_score": 1,
            "tags": 1,
            "developer": 1,
            "release_year": 1
        }
    ).limit(limit))
    
    return {
        "query": q,
        "count": len(games),
        "games": convert_mongo_data(games)
    }

@app.post("/games/filter")
def filter_games(filter_request: FilterRequest):
    """Filter games by multiple criteria"""
    query = {}
    
    # Price filter
    query["original_price"] = {"$gte": filter_request.min_price, "$lte": filter_request.max_price}
    
    # Sentiment filter
    if filter_request.min_sentiment > 0:
        query["overall_sentiment_score"] = {"$gte": filter_request.min_sentiment}
    
    # Popularity filter
    if filter_request.min_popularity > 0:
        query["popularity_score"] = {"$gte": filter_request.min_popularity}
    
    # Tags filter
    if filter_request.tags:
        query["tags"] = {"$in": [tag.lower() for tag in filter_request.tags]}
    
    # Categories filter
    if filter_request.categories:
        query["categories"] = {"$in": [cat.lower() for cat in filter_request.categories]}
    
    games = list(db.steam_games.find(
        query,
        {
            "_id": 0,
            "title": 1,
            "original_price": 1,
            "overall_sentiment_score": 1,
            "popularity_score": 1,
            "tags": 1,
            "developer": 1,
            "release_year": 1,
            "categories": 1
        }
    ).sort("popularity_score", -1).limit(filter_request.limit))
    
    return {
        "filters": filter_request.dict(),
        "count": len(games),
        "games": convert_mongo_data(games)
    }

@app.post("/recommend/preferences")
def recommend_by_preferences(request: RecommendationRequest):
    """Get personalized recommendations based on user preferences"""
    try:
        recommendations = recommender.recommend_by_preferences(
            request.preferences.dict(),
            request.limit
        )
        
        if not recommendations:
            return {
                "recommendations": [],
                "message": "No games match your preferences. Try relaxing your criteria."
            }
        
        formatted_recommendations = []
        for rec in recommendations:
            game = rec['game']
            explanations = recommender.get_explanation(game, rec['score_breakdown'])
            
            formatted_recommendations.append({
                "title": game.get("title", "Unknown"),
                "score": round(rec['score'], 3),
                "price": game.get("original_price", 0),
                "sentiment": game.get("overall_sentiment_score", 0.5),
                "popularity": game.get("popularity_score", 0.3),
                "tags": game.get("tags", [])[:5],
                "categories": game.get("categories", []),
                "developer": game.get("developer", "Unknown"),
                "release_year": game.get("release_year"),
                "explanations": explanations,
                "score_breakdown": rec['score_breakdown']
            })
        
        return {
            "recommendations": formatted_recommendations,
            "count": len(formatted_recommendations),
            "preferences_used": request.preferences.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/recommend/similar/{game_title}")
def recommend_similar_games(
    game_title: str,
    limit: int = Query(5, ge=1, le=20)
):
    """Get games similar to a specific game"""
    try:
        recommendations = recommender.recommend_by_game(game_title, limit)
        
        if not recommendations:
            # Fallback: find games with similar tags
            game = db.steam_games.find_one({"title_lower": game_title.lower()})
            if not game:
                raise HTTPException(status_code=404, detail=f"Game '{game_title}' not found")
            
            # Find games with common tags
            common_tags = game.get('tags', [])[:3]
            if common_tags:
                similar_games = list(db.steam_games.find(
                    {
                        "title_lower": {"$ne": game_title.lower()},
                        "tags": {"$in": common_tags}
                    },
                    {"_id": 0}
                ).limit(limit))
                
                recommendations = [
                    {"game": g, "similarity_score": 0.5} 
                    for g in similar_games
                ]
        
        formatted_recommendations = []
        for rec in recommendations:
            game = rec['game']
            formatted_recommendations.append({
                "title": game.get("title", "Unknown"),
                "similarity_score": round(rec['similarity_score'], 3),
                "price": game.get("original_price", 0),
                "sentiment": game.get("overall_sentiment_score", 0.5),
                "tags": game.get("tags", [])[:5],
                "developer": game.get("developer", "Unknown")
            })
        
        return {
            "source_game": game_title,
            "recommendations": formatted_recommendations,
            "count": len(formatted_recommendations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar games: {str(e)}")

@app.get("/stats")
def get_stats():
    """Get database statistics"""
    total_games = db.steam_games.count_documents({})
    
    # Price statistics
    price_stats = list(db.steam_games.aggregate([
        {"$group": {
            "_id": None,
            "avg_price": {"$avg": "$original_price"},
            "max_price": {"$max": "$original_price"},
            "min_price": {"$min": "$original_price"},
            "free_games": {"$sum": {"$cond": [{"$eq": ["$original_price", 0]}, 1, 0]}}
        }}
    ]))
    
    # Tag statistics
    top_tags = list(db.steam_games.aggregate([
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]))
    
    # Sentiment distribution
    sentiment_stats = list(db.steam_games.aggregate([
        {"$bucket": {
            "groupBy": "$overall_sentiment_score",
            "boundaries": [0, 0.3, 0.5, 0.7, 0.9, 1.0],
            "default": "other",
            "output": {
                "count": {"$sum": 1}
            }
        }}
    ]))
    
    return {
        "total_games": total_games,
        "price_statistics": price_stats[0] if price_stats else {},
        "top_tags": top_tags,
        "sentiment_distribution": sentiment_stats
    }

@app.get("/tags")
def get_all_tags(limit: int = Query(50, ge=1, le=100)):
    """Get all unique tags"""
    tags = db.steam_games.distinct("tags")
    return {"tags": sorted(tags)[:limit], "count": len(tags)}

@app.get("/categories")
def get_all_categories():
    """Get all unique categories"""
    categories = db.steam_games.distinct("categories")
    return {"categories": sorted(categories), "count": len(categories)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)