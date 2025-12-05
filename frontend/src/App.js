import React, { useState, useEffect } from 'react';
import { gameAPI } from './services/api';
import './App.css';

function App() {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('Checking backend connection...');
  const [recommendations, setRecommendations] = useState([]);
  const [showRecommendations, setShowRecommendations] = useState(false);
  
  // User preferences state
  const [preferences, setPreferences] = useState({
    max_price: 50,
    preferred_tags: ['action', 'rpg', 'adventure'],
    min_sentiment: 0.7,
    min_popularity: 0.5,
    system_specs: {
      memory_gb: 8,
      os_type: 'windows'
    }
  });

  useEffect(() => {
    testConnection();
    loadGames();
  }, []);

  const testConnection = async () => {
    try {
      const response = await gameAPI.healthCheck();
      setConnectionStatus(`✅ Connected to backend (${response.data.games_loaded} games loaded)`);
    } catch (error) {
      setConnectionStatus('❌ Cannot connect to backend. Make sure FastAPI is running on http://localhost:8000');
    }
  };

  const loadGames = async () => {
    try {
      setLoading(true);
      const response = await gameAPI.getGames(1, 12);
      setGames(response.data.games);
      setShowRecommendations(false);
    } catch (error) {
      console.error('Error loading games:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!search.trim()) {
      loadGames();
      return;
    }
    
    try {
      setLoading(true);
      const response = await gameAPI.searchGames(search);
      setGames(response.data.games);
      setShowRecommendations(false);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRecommendations = async () => {
    try {
      setLoading(true);
      const response = await gameAPI.getRecommendations(preferences);
      setRecommendations(response.data.recommendations);
      setShowRecommendations(true);
      console.log('Recommendations:', response.data);
    } catch (error) {
      console.error('Recommendation error:', error);
    } finally {
      setLoading(false);
    }
  };

  const updatePreference = (key, value) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const addTag = (tag) => {
    if (tag && !preferences.preferred_tags.includes(tag)) {
      setPreferences(prev => ({
        ...prev,
        preferred_tags: [...prev.preferred_tags, tag]
      }));
    }
  };

  const removeTag = (tagToRemove) => {
    setPreferences(prev => ({
      ...prev,
      preferred_tags: prev.preferred_tags.filter(tag => tag !== tagToRemove)
    }));
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🎮 Game Recommendation System</h1>
          <p className="subtitle">Knowledge-Based Recommendations using FastAPI + React + MongoDB</p>
          <div className="connection-status">{connectionStatus}</div>
        </div>
        
        <div className="search-section">
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              placeholder="Search games by title..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="search-input"
            />
            <button type="submit" className="btn btn-search">Search</button>
            <button type="button" onClick={loadGames} className="btn btn-reset">Show All</button>
          </form>
          
          <div className="preferences-section">
            <h3>Your Preferences:</h3>
            <div className="preferences-grid">
              <div className="preference-item">
                <label>Max Price: ${preferences.max_price}</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={preferences.max_price}
                  onChange={(e) => updatePreference('max_price', parseInt(e.target.value))}
                />
              </div>
              
              <div className="preference-item">
                <label>Min Sentiment: {(preferences.min_sentiment * 100).toFixed(0)}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={preferences.min_sentiment * 100}
                  onChange={(e) => updatePreference('min_sentiment', parseInt(e.target.value) / 100)}
                />
              </div>
              
              <div className="preference-item">
                <label>Min Popularity: {(preferences.min_popularity * 100).toFixed(0)}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={preferences.min_popularity * 100}
                  onChange={(e) => updatePreference('min_popularity', parseInt(e.target.value) / 100)}
                />
              </div>
              
              <div className="preference-item tags-input">
                <label>Preferred Tags:</label>
                <div className="tags-container">
                  {preferences.preferred_tags.map(tag => (
                    <span key={tag} className="tag">
                      {tag}
                      <button onClick={() => removeTag(tag)} className="tag-remove">×</button>
                    </span>
                  ))}
                  <input
                    type="text"
                    placeholder="Add tag..."
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        addTag(e.target.value.toLowerCase());
                        e.target.value = '';
                      }
                    }}
                    className="tag-input"
                  />
                </div>
              </div>
            </div>
            
            <button onClick={getRecommendations} className="btn btn-recommend">
              🎯 Get Personalized Recommendations
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        {loading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading...</p>
          </div>
        ) : showRecommendations ? (
          <div className="recommendations-section">
            <h2>🎯 Personalized Recommendations ({recommendations.length})</h2>
            <p className="section-subtitle">Based on your preferences</p>
            
            <div className="games-grid">
              {recommendations.map((rec, index) => (
                <div key={index} className="game-card recommendation-card">
                  <div className="game-card-header">
                    <h3 className="game-title">{rec.title}</h3>
                    <span className="recommendation-score">Match: {(rec.score * 100).toFixed(1)}%</span>
                  </div>
                  
                  <div className="game-info">
                    <div className="info-item">
                      <span className="info-label">Price:</span>
                      <span className={`info-value ${rec.price === 0 ? 'free' : ''}`}>
                        ${rec.price.toFixed(2)}
                      </span>
                    </div>
                    
                    <div className="info-item">
                      <span className="info-label">Sentiment:</span>
                      <span className="info-value sentiment">
                        {(rec.sentiment * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="info-item">
                      <span className="info-label">Popularity:</span>
                      <span className="info-value popularity">
                        {(rec.popularity * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    <div className="info-item">
                      <span className="info-label">Developer:</span>
                      <span className="info-value">{rec.developer}</span>
                    </div>
                  </div>
                  
                  {rec.explanations && rec.explanations.length > 0 && (
                    <div className="explanations">
                      <strong>Why recommended:</strong>
                      <ul>
                        {rec.explanations.map((exp, i) => (
                          <li key={i}>✓ {exp}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {rec.tags && rec.tags.length > 0 && (
                    <div className="game-tags">
                      {rec.tags.slice(0, 5).map((tag, i) => (
                        <span key={i} className="tag">{tag}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
            
            <button onClick={() => setShowRecommendations(false)} className="btn btn-back">
              ← Back to All Games
            </button>
          </div>
        ) : games.length === 0 ? (
          <div className="empty-state">
            <h2>No games found</h2>
            <p>Try a different search or adjust your preferences</p>
            <button onClick={loadGames} className="btn btn-primary">Show All Games</button>
          </div>
        ) : (
          <div className="games-section">
            <div className="section-header">
              <h2>Games ({games.length})</h2>
              <button onClick={getRecommendations} className="btn btn-recommend-small">
                Get Recommendations
              </button>
            </div>
            
            <div className="games-grid">
              {games.map((game, index) => (
                <div key={index} className="game-card">
                  <h3 className="game-title">{game.title}</h3>
                  
                  <div className="game-meta">
                    <span className="game-price">${game.original_price?.toFixed(2) || '0.00'}</span>
                    <span className="game-developer">{game.developer || 'Unknown'}</span>
                    {game.release_year && (
                      <span className="game-year">{game.release_year}</span>
                    )}
                  </div>
                  
                  <div className="game-scores">
                    <div className="score-item">
                      <span className="score-label">Sentiment:</span>
                      <span className="score-value sentiment">
                        {((game.overall_sentiment_score || 0.5) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="score-item">
                      <span className="score-label">Popularity:</span>
                      <span className="score-value popularity">
                        {((game.popularity_score || 0.3) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  
                  {game.tags && game.tags.length > 0 && (
                    <div className="game-tags">
                      {game.tags.slice(0, 3).map((tag, i) => (
                        <span key={i} className="tag">{tag}</span>
                      ))}
                      {game.tags.length > 3 && (
                        <span className="tag-more">+{game.tags.length - 3} more</span>
                      )}
                    </div>
                  )}
                  
                  {game.categories && game.categories.length > 0 && (
                    <div className="game-categories">
                      {game.categories.map((cat, i) => (
                        <span key={i} className="category">{cat}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Game Recommendation System • Knowledge-Based Recommendations • Built with FastAPI, React, and MongoDB</p>
        <p className="footer-tech">Backend: FastAPI (Python) • Frontend: React • Database: MongoDB • ML: Scikit-learn</p>
      </footer>
    </div>
  );
}

export default App;