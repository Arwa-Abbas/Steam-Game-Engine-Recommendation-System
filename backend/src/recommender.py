import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
import re

class GameRecommender:
    """Knowledge-based recommendation system for games"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.games = []
        self.game_features = []
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = MinMaxScaler()
        
    def load_games(self):
        """Load games from MongoDB"""
        self.games = list(self.db.steam_games.find({}))
        print(f"📊 Loaded {len(self.games)} games for recommendation system")
        return self.games
    
    def prepare_features(self):
        """Prepare feature vectors for similarity calculation"""
        if not self.games:
            self.load_games()
        
        feature_strings = []
        
        for game in self.games:
            # Initialize features list for each game
            features = []
            
            # Add tags
            if game.get('tags'):
                features.extend([tag.lower() for tag in game['tags']])
            
            # Add categories
            if game.get('categories'):
                features.extend([cat.lower() for cat in game['categories']])
            
            # Add game features
            if game.get('features'):
                features.extend([feat.lower() for feat in game['features']])
            
            # Add keywords from description
            if game.get('description_keywords'):
                features.extend([kw.lower() for kw in game['description_keywords']])
            
            # Add technical features
            if game.get('os_type'):
                features.append(f"os_{game['os_type'].lower()}")
            if game.get('gpu_brand'):
                features.append(f"gpu_{game['gpu_brand'].lower()}")
            if game.get('cpu_brand'):
                features.append(f"cpu_{game['cpu_brand'].lower()}")
            if game.get('price_category'):
                features.append(f"price_{game['price_category'].lower()}")
            
            # Add price bucket for better matching
            price = game.get('original_price', 0)
            if price == 0:
                features.append("price_free")
            elif price < 10:
                features.append("price_cheap")
            elif price < 30:
                features.append("price_mid")
            elif price < 60:
                features.append("price_expensive")
            else:
                features.append("price_premium")
            
            # Create feature string
            feature_string = ' '.join(features)
            feature_strings.append(feature_string)
        
        # Create TF-IDF vectors
        if feature_strings:
            self.game_features = self.tfidf_vectorizer.fit_transform(feature_strings).toarray()
            print(f"✅ Created TF-IDF matrix with shape: {self.game_features.shape}")
        else:
            print("⚠️ No features generated!")
            self.game_features = np.array([])
        
        return self.game_features
    
    def calculate_similarity_matrix(self):
        """Calculate cosine similarity between all games"""
        if not hasattr(self, 'game_features') or self.game_features.size == 0:
            self.prepare_features()
        
        if self.game_features.size == 0:
            print("⚠️ No features available for similarity calculation")
            return np.array([])
        
        similarity_matrix = cosine_similarity(self.game_features)
        print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix
    
    def recommend_by_game(self, game_title: str, top_n: int = 5):
        """Recommend games similar to a specific game"""
        # Find the game index
        game_index = None
        for i, game in enumerate(self.games):
            if game['title'].lower() == game_title.lower():
                game_index = i
                break
        
        if game_index is None:
            print(f"⚠️ Game '{game_title}' not found in database")
            return []
        
        # Get similarity scores
        similarity_matrix = self.calculate_similarity_matrix()
        
        if similarity_matrix.size == 0:
            return []
        
        similar_indices = np.argsort(similarity_matrix[game_index])[::-1][1:top_n+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'game': self.games[idx],
                'similarity_score': float(similarity_matrix[game_index][idx])
            })
        
        return recommendations
    
    def recommend_by_preferences(self, user_preferences: Dict[str, Any], top_n: int = 10):
        """
        Knowledge-based recommendations based on user preferences
        """
        if not self.games:
            self.load_games()
        
        print(f"🎯 Getting recommendations for preferences: {user_preferences}")
        
        scored_games = []
        
        for game in self.games:
            score = self.calculate_preference_score(game, user_preferences)
            
            # Only include games that meet minimum requirements
            if score['total_score'] > 0.1:  # Lower threshold
                scored_games.append({
                    'game': game,
                    'score': score['total_score'],
                    'score_breakdown': score
                })
        
        # Sort by total score
        scored_games.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"📈 Found {len(scored_games)} games meeting criteria")
        
        return scored_games[:top_n]
    
    def calculate_preference_score(self, game: Dict, preferences: Dict) -> Dict:
        """
        Calculate preference match score with detailed breakdown
        """
        scores = {
            'tag_match': 0,
            'category_match': 0,
            'price_match': 0,
            'system_match': 0,
            'sentiment_score': 0,
            'popularity_score': 0,
            'total_score': 0
        }
        
        weights = {
            'tag_match': 0.30,
            'category_match': 0.20,
            'price_match': 0.25,
            'system_match': 0.10,
            'sentiment_score': 0.10,
            'popularity_score': 0.05
        }
        
        # 1. TAG MATCHING (30%)
        if preferences.get('preferred_tags'):
            user_tags = set([t.lower() for t in preferences['preferred_tags']])
            game_tags = set([t.lower() for t in game.get('tags', [])])
            
            if user_tags and game_tags:
                tag_intersection = len(user_tags.intersection(game_tags))
                scores['tag_match'] = min(tag_intersection / max(len(user_tags), 1), 1.0)
        
        # 2. CATEGORY MATCHING (20%)
        if preferences.get('preferred_categories'):
            user_cats = set([c.lower() for c in preferences['preferred_categories']])
            game_cats = set([c.lower() for c in game.get('categories', [])])
            
            if user_cats and game_cats:
                cat_intersection = len(user_cats.intersection(game_cats))
                scores['category_match'] = min(cat_intersection / max(len(user_cats), 1), 1.0)
        
        # 3. PRICE MATCHING - FIXED! (25%)
        max_price = preferences.get('max_price', 100)
        game_price = game.get('original_price', 0)
        
        if max_price > 0:  # Only calculate if max_price is valid
            if game_price <= max_price:
                # Normalize: cheaper games get higher scores
                if max_price > 0:
                    scores['price_match'] = 1.0 - (game_price / max_price)
                else:
                    scores['price_match'] = 1.0
            else:
                scores['price_match'] = 0
        
        # 4. System requirements matching (10%)
        if preferences.get('system_specs'):
            system_match = self.check_system_compatibility(game, preferences['system_specs'])
            scores['system_match'] = 1.0 if system_match else 0
        else:
            scores['system_match'] = 1.0  # If no specs provided, assume compatible
        
        # 5. Sentiment score (10%)
        sentiment = game.get('overall_sentiment_score', 0.5)
        min_sentiment = preferences.get('min_sentiment', 0.0)
        if sentiment >= min_sentiment:
            # Normalize sentiment score
            scores['sentiment_score'] = sentiment
        else:
            scores['sentiment_score'] = 0
        
        # 6. Popularity score (5%)
        popularity = game.get('popularity_score', 0.3)
        min_popularity = preferences.get('min_popularity', 0.0)
        if popularity >= min_popularity:
            scores['popularity_score'] = popularity
        else:
            scores['popularity_score'] = 0
        
        # Calculate weighted total score
        total_score = 0
        for key, weight in weights.items():
            total_score += scores[key] * weight
        
        scores['total_score'] = min(total_score, 1.0)
        
        return scores
    
    def check_system_compatibility(self, game: Dict, user_specs: Dict) -> bool:
        """Check if game requirements match user's system"""
        # If no specs provided, assume compatible
        if not user_specs:
            return True
        
        # Check memory
        if 'memory_gb' in user_specs and game.get('memory_gb'):
            if game['memory_gb'] > user_specs['memory_gb']:
                return False
        
        # Check OS - only fail if specifically incompatible
        if 'os_type' in user_specs and game.get('os_type'):
            user_os = user_specs['os_type'].lower()
            game_os = game['os_type'].lower()
            
            # Windows can't run Mac/Linux games
            if user_os == 'windows' and game_os in ['mac', 'linux']:
                return False
        
        # Check storage (optional)
        if 'storage_gb' in user_specs and game.get('storage_gb'):
            if game['storage_gb'] > user_specs['storage_gb']:
                return False
        
        return True
    
    def get_explanation(self, game: Dict, score_breakdown: Dict) -> List[str]:
        """Generate explanation for why a game was recommended"""
        explanations = []
        
        if score_breakdown['tag_match'] > 0.8:
            explanations.append("Perfect match for your tag preferences")
        elif score_breakdown['tag_match'] > 0.5:
            explanations.append("Good match with your interests")
        elif score_breakdown['tag_match'] > 0.2:
            explanations.append("Some matching tags found")
        
        if score_breakdown['category_match'] > 0.5:
            explanations.append("Matches your preferred categories")
        
        if score_breakdown['price_match'] > 0.9:
            explanations.append("Excellent value within your budget")
        elif score_breakdown['price_match'] > 0.7:
            explanations.append("Good price for your budget")
        elif game.get('original_price', 0) == 0:
            explanations.append("Free to play - no cost")
        
        if game.get('overall_sentiment_score', 0) > 0.85:
            explanations.append("Exceptional player ratings")
        elif game.get('overall_sentiment_score', 0) > 0.75:
            explanations.append("Highly rated by the community")
        
        if game.get('popularity_score', 0) > 0.8:
            explanations.append("Very popular choice among players")
        
        return explanations[:3]  # Return top 3 explanations