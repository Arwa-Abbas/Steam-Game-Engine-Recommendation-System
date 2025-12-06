import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any, Optional
import re

class GameRecommender:
    """Enhanced knowledge-based recommendation system for games"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.games = []
        self.game_features = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.scaler = MinMaxScaler()
        self.similarity_matrix = None
        self.model_path = "models/recommender_model.pkl"
        
    def load_games(self):
        """Load games from MongoDB"""
        self.games = list(self.db.steam_games.find({}))
        print(f"📊 Loaded {len(self.games)} games for recommendation system")
        return self.games
    
    def prepare_features(self):
        """Prepare enhanced feature vectors for similarity calculation"""
        if not self.games:
            self.load_games()
        
        feature_strings = []
        
        for game in self.games:
            features = []
            
            # Tags (high weight - repeat 3 times)
            if game.get('tags'):
                tags = [tag.lower() for tag in game['tags']]
                features.extend(tags * 3)
            
            # Categories (medium weight - repeat 2 times)
            if game.get('categories'):
                cats = [cat.lower() for cat in game['categories']]
                features.extend(cats * 2)
            
            # Features (medium weight)
            if game.get('features'):
                feats = [feat.lower() for feat in game['features']]
                features.extend(feats * 2)
            
            # Languages (important for accessibility)
            if game.get('languages'):
                langs = [f"lang_{lang.lower()}" for lang in game['languages']]
                features.extend(langs)
            
            # Developer and Publisher
            if game.get('developer'):
                features.append(f"dev_{game['developer'].lower()}")
            if game.get('publisher'):
                features.append(f"pub_{game['publisher'].lower()}")
            
            # Price category (exact match for budget)
            price = game.get('discounted_price', game.get('original_price', 0))
            if price == 0:
                features.extend(["price_free", "price_0"] * 3)  # High weight for free
            elif price <= 5:
                features.extend(["price_budget", "price_5"] * 2)
            elif price <= 10:
                features.extend(["price_budget", "price_10"] * 2)
            elif price <= 20:
                features.extend(["price_mid", "price_20"])
            elif price <= 30:
                features.extend(["price_mid", "price_30"])
            elif price <= 50:
                features.extend(["price_premium", "price_50"])
            else:
                features.extend(["price_expensive", "price_60plus"])
            
            # System requirements (OS, specs)
            if game.get('os_type'):
                features.extend([f"os_{game['os_type'].lower()}"] * 2)
            if game.get('memory_gb'):
                mem = game['memory_gb']
                features.append(f"ram_{mem}gb")
                if mem <= 4:
                    features.append("ram_low")
                elif mem <= 8:
                    features.append("ram_mid")
                else:
                    features.append("ram_high")
            
            # Quality indicators (reviews/sentiment)
            reviews = game.get('all_reviews_count', 0)
            if reviews > 10000:
                features.extend(["popular", "well_reviewed"])
            
            sentiment = game.get('overall_sentiment_score', 0.5)
            if sentiment >= 0.8:
                features.extend(["highly_rated", "positive_reviews"] * 2)
            elif sentiment >= 0.7:
                features.append("positive_reviews")
            
            # Keywords from description
            if game.get('description_keywords'):
                keywords = [kw.lower() for kw in game['description_keywords'][:10]]
                features.extend(keywords)
            
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
        """Calculate and cache similarity matrix"""
        if self.game_features is None or self.game_features.size == 0:
            self.prepare_features()
        
        if self.game_features.size == 0:
            print("⚠️ No features available for similarity calculation")
            return np.array([])
        
        self.similarity_matrix = cosine_similarity(self.game_features)
        print(f"✅ Similarity matrix shape: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def train_and_save_model(self):
        """Train the model and save for quick loading"""
        print("🎓 Training recommendation model...")
        
        self.load_games()
        self.prepare_features()
        self.calculate_similarity_matrix()
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Save model components
        model_data = {
            'games': self.games,
            'game_features': self.game_features,
            'similarity_matrix': self.similarity_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {self.model_path}")
        return True
    
    def load_model(self):
        """Load pre-trained model"""
        if not os.path.exists(self.model_path):
            print("⚠️ No saved model found. Training new model...")
            return self.train_and_save_model()
        
        try:
            print("📂 Loading pre-trained model...")
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.games = model_data['games']
            self.game_features = model_data['game_features']
            self.similarity_matrix = model_data['similarity_matrix']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.scaler = model_data['scaler']
            
            print(f"✅ Model loaded successfully ({len(self.games)} games)")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Training new model...")
            return self.train_and_save_model()
    
    def recommend_by_preferences(self, user_preferences: Dict[str, Any], top_n: int = 10):
        """
        Knowledge-based recommendations with strict filtering
        """
        if not self.games:
            self.load_games()
        
        print(f"🎯 Finding games for preferences: {user_preferences}")
        
        # STRICT FILTERING FIRST
        filtered_games = []
        
        for game in self.games:
            # 1. PRICE FILTER (STRICT - use discounted_price)
            max_price = user_preferences.get('max_price', 1000)
            game_price = game.get('discounted_price', game.get('original_price', 0))
            
            if game_price > max_price:
                continue
            
            # 2. LANGUAGE FILTER (if specified)
            required_languages = user_preferences.get('languages', [])
            if required_languages:
                game_languages = [lang.lower() for lang in game.get('languages', [])]
                if not any(lang.lower() in game_languages for lang in required_languages):
                    continue
            
            # 3. SYSTEM REQUIREMENTS FILTER
            system_specs = user_preferences.get('system_specs', {})
            if system_specs:
                # Check memory
                if system_specs.get('memory_gb') and game.get('memory_gb'):
                    if game['memory_gb'] > system_specs['memory_gb']:
                        continue
                
                # Check OS compatibility
                if system_specs.get('os_type') and game.get('os_type'):
                    user_os = system_specs['os_type'].lower()
                    game_os = game['os_type'].lower()
                    if user_os == 'windows' and game_os in ['mac', 'linux']:
                        continue
                    elif user_os == 'mac' and game_os not in ['mac']:
                        continue
                    elif user_os == 'linux' and game_os not in ['linux']:
                        continue
                
                # Check storage
                if system_specs.get('storage_gb') and game.get('storage_gb'):
                    if game['storage_gb'] > system_specs['storage_gb']:
                        continue
            
            # 4. DEVELOPER/PUBLISHER FILTER
            preferred_devs = user_preferences.get('developers', [])
            preferred_pubs = user_preferences.get('publishers', [])
            
            if preferred_devs:
                game_dev = game.get('developer', '').lower()
                if not any(dev.lower() in game_dev for dev in preferred_devs):
                    continue
            
            if preferred_pubs:
                game_pub = game.get('publisher', '').lower()
                if not any(pub.lower() in game_pub for pub in preferred_pubs):
                    continue
            
            # Game passed all filters
            filtered_games.append(game)
        
        print(f"📊 {len(filtered_games)} games passed filters")
        
        if not filtered_games:
            return []
        
        # NOW SCORE THE FILTERED GAMES
        scored_games = []
        
        for game in filtered_games:
            score = self.calculate_preference_score(game, user_preferences)
            
            scored_games.append({
                'game': game,
                'score': score['total_score'],
                'score_breakdown': score
            })
        
        # Sort by score (reviews + tags)
        scored_games.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✅ Returning top {min(top_n, len(scored_games))} recommendations")
        
        return scored_games[:top_n]
    
    def calculate_preference_score(self, game: Dict, preferences: Dict) -> Dict:
        """
        Calculate match score based on reviews, tags, and features
        """
        scores = {
            'tag_match': 0,
            'review_quality': 0,
            'popularity': 0,
            'total_score': 0
        }
        
        # 1. TAG MATCHING (50% weight)
        preferred_tags = preferences.get('preferred_tags', [])
        if preferred_tags:
            user_tags = set([t.lower() for t in preferred_tags])
            game_tags = set([t.lower() for t in game.get('tags', [])])
            
            if user_tags and game_tags:
                matches = len(user_tags.intersection(game_tags))
                scores['tag_match'] = min(matches / len(user_tags), 1.0)
        else:
            scores['tag_match'] = 0.5  # Neutral if no tags specified
        
        # 2. REVIEW QUALITY (30% weight)
        # Prioritize games with good reviews
        sentiment = game.get('overall_sentiment_score', 0.5)
        review_count = game.get('all_reviews_count', 0)
        
        # Normalize review count (log scale)
        import math
        if review_count > 0:
            review_score = min(math.log10(review_count + 1) / 5, 1.0)  # Log scale
        else:
            review_score = 0
        
        # Combine sentiment and review count
        scores['review_quality'] = (sentiment * 0.7) + (review_score * 0.3)
        
        # 3. POPULARITY (20% weight)
        popularity = game.get('popularity_score', 0.3)
        scores['popularity'] = popularity
        
        # TOTAL SCORE (weighted)
        weights = {
            'tag_match': 0.50,
            'review_quality': 0.30,
            'popularity': 0.20
        }
        
        total = 0
        for key, weight in weights.items():
            total += scores[key] * weight
        
        scores['total_score'] = min(total, 1.0)
        
        return scores
    
    def recommend_similar_games(self, game_title: str, top_n: int = 5):
        """Find similar games using pre-computed similarity matrix"""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        # Find game index
        game_index = None
        for i, game in enumerate(self.games):
            if game['title'].lower() == game_title.lower():
                game_index = i
                break
        
        if game_index is None:
            return []
        
        # Get similar games
        similar_indices = np.argsort(self.similarity_matrix[game_index])[::-1][1:top_n+1]
        
        recommendations = []
        for idx in similar_indices:
            recommendations.append({
                'game': self.games[idx],
                'similarity_score': float(self.similarity_matrix[game_index][idx])
            })
        
        return recommendations
    
    def get_explanation(self, game: Dict, score_breakdown: Dict) -> List[str]:
        """Generate explanations for recommendations"""
        explanations = []
        
        # Tag match
        if score_breakdown['tag_match'] > 0.7:
            explanations.append("Perfect match for your interests")
        elif score_breakdown['tag_match'] > 0.4:
            explanations.append("Matches several of your preferences")
        
        # Review quality
        sentiment = game.get('overall_sentiment_score', 0.5)
        reviews = game.get('all_reviews_count', 0)
        
        if sentiment >= 0.85 and reviews > 1000:
            explanations.append("Highly rated with lots of positive reviews")
        elif sentiment >= 0.75:
            explanations.append("Well-reviewed by players")
        
        # Price
        price = game.get('discounted_price', game.get('original_price', 0))
        if price == 0:
            explanations.append("Free to play")
        elif price < 10:
            explanations.append("Great value for money")
        
        # Popularity
        if game.get('popularity_score', 0) > 0.7:
            explanations.append("Very popular among players")
        
        return explanations[:3]