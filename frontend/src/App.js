import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('explore');
  const [games, setGames] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Available options
  const [availableTags, setAvailableTags] = useState([]);
  const [availableLanguages, setAvailableLanguages] = useState([]);
  
  // User preferences
  const [preferences, setPreferences] = useState({
    max_price: 50,
    preferred_tags: [],
    languages: [],
    developers: [],
    publishers: [],
    system_specs: {
      memory_gb: null,
      storage_gb: null,
      os_type: ''
    }
  });
  
  const [tagInput, setTagInput] = useState('');
  const [showFilters, setShowFilters] = useState(false);

  // Utility function to extract Steam app ID from URL
  const extractSteamAppId = (url) => {
    if (!url || typeof url !== 'string') return null;
    
    try {
      const patterns = [
        /store\.steampowered\.com\/app\/(\d+)/,
        /\/app\/(\d+)/,
        /appid=(\d+)/,
        /\/(\d+)\/?$/
      ];
      
      for (const pattern of patterns) {
        const match = url.match(pattern);
        if (match && match[1]) {
          return match[1];
        }
      }
      
      return null;
    } catch (error) {
      console.error('Error extracting app ID:', error);
      return null;
    }
  };

  // Function to get game poster URL
  const getGamePosterUrl = (game) => {
    if (!game) return null;
    
    const appId = extractSteamAppId(game.link);
    
    if (appId) {
      return `https://cdn.cloudflare.steamstatic.com/steam/apps/${appId}/header.jpg`;
    }
    
    return null;
  };

  // Function to create a placeholder image
  const getPlaceholderImage = (title) => {
    if (!title) return '';
    
    const colors = ['8b5cf6', 'ec4899', '06b6d4', '10b981', 'f59e0b'];
    let hash = 0;
    for (let i = 0; i < title.length; i++) {
      hash = title.charCodeAt(i) + ((hash << 5) - hash);
    }
    const colorIndex = Math.abs(hash) % colors.length;
    const color = colors[colorIndex];
    
    const shortTitle = title.length > 30 ? title.substring(0, 27) + '...' : title;
    const encodedTitle = encodeURIComponent(shortTitle);
    
    return `https://via.placeholder.com/460x215/${color}/ffffff?text=${encodedTitle}`;
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // Load stats
      const statsRes = await axios.get(`${API_BASE}/stats`);
      setStats(statsRes.data);
      
      // Load available tags
      const tagsRes = await axios.get(`${API_BASE}/tags`);
      setAvailableTags(tagsRes.data.tags.slice(0, 50));
      
      // Load available languages
      const langsRes = await axios.get(`${API_BASE}/languages`);
      setAvailableLanguages(langsRes.data.languages);
      
      // Load games
      loadGames();
    } catch (error) {
      console.error('Error loading initial data:', error);
    }
  };

  const loadGames = async (page = 1) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/games`, {
        params: { page, limit: 12 }
      });
      setGames(response.data.games);
    } catch (error) {
      console.error('Error loading games:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      loadGames();
      return;
    }
    
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE}/games/search`, {
        params: { q: searchQuery }
      });
      setGames(response.data.games);
      setActiveTab('explore');
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRecommendations = async () => {
    try {
      setLoading(true);
      
      // Clean preferences
      const cleanPrefs = {
        ...preferences,
        system_specs: preferences.system_specs.os_type ? preferences.system_specs : null
      };
      
      const response = await axios.post(`${API_BASE}/recommend`, {
        preferences: cleanPrefs,
        limit: 12
      });
      
      setRecommendations(response.data.recommendations);
      setActiveTab('recommendations');
    } catch (error) {
      console.error('Recommendation error:', error);
      alert('Error getting recommendations. Please try adjusting your preferences.');
    } finally {
      setLoading(false);
    }
  };

  const addTag = (tag) => {
    if (tag && !preferences.preferred_tags.includes(tag)) {
      setPreferences(prev => ({
        ...prev,
        preferred_tags: [...prev.preferred_tags, tag.toLowerCase()]
      }));
    }
    setTagInput('');
  };

  const removeTag = (tag) => {
    setPreferences(prev => ({
      ...prev,
      preferred_tags: prev.preferred_tags.filter(t => t !== tag)
    }));
  };

  const addLanguage = (lang) => {
    if (lang && !preferences.languages.includes(lang)) {
      setPreferences(prev => ({
        ...prev,
        languages: [...prev.languages, lang]
      }));
    }
  };

  const removeLanguage = (lang) => {
    setPreferences(prev => ({
      ...prev,
      languages: prev.languages.filter(l => l !== lang)
    }));
  };

  const updateSystemSpec = (key, value) => {
    setPreferences(prev => ({
      ...prev,
      system_specs: {
        ...prev.system_specs,
        [key]: value || null
      }
    }));
  };

  return (
    <div className="app">
      {/* Animated Background */}
      <div className="animated-bg">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🎮</span>
            <h1>GameFinder</h1>
          </div>
          
          <nav className="nav-tabs">
            <button 
              className={`nav-tab ${activeTab === 'explore' ? 'active' : ''}`}
              onClick={() => setActiveTab('explore')}
            >
              Explore
            </button>
            <button 
              className={`nav-tab ${activeTab === 'recommendations' ? 'active' : ''}`}
              onClick={() => setActiveTab('recommendations')}
            >
              For You
            </button>
            <button 
              className={`nav-tab ${activeTab === 'stats' ? 'active' : ''}`}
              onClick={() => setActiveTab('stats')}
            >
              Stats
            </button>
          </nav>
          
          <form onSubmit={handleSearch} className="search-bar">
            <input
              type="text"
              placeholder="Search games..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            <button type="submit" className="search-btn">
              <span>🔍</span>
            </button>
          </form>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        
        {/* Explore Tab */}
        {activeTab === 'explore' && (
          <div className="tab-content fade-in">
            <div className="section-header">
              <h2>Discover Games</h2>
              <button 
                className="filter-toggle"
                onClick={() => setShowFilters(!showFilters)}
              >
                {showFilters ? '✕ Close' : 'Filters'}
              </button>
            </div>
            
            {showFilters && (
              <div className="filters-panel slide-down">
                <div className="filter-grid">
                  <div className="filter-group">
                    <label>Max Price: ${preferences.max_price}</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={preferences.max_price}
                      onChange={(e) => setPreferences(prev => ({
                        ...prev,
                        max_price: parseInt(e.target.value)
                      }))}
                      className="slider"
                    />
                  </div>
                  
                  <div className="filter-group">
                    <label>Tags</label>
                    <div className="tags-input">
                      {preferences.preferred_tags.map(tag => (
                        <span key={tag} className="tag">
                          {tag}
                          <button onClick={() => removeTag(tag)}>×</button>
                        </span>
                      ))}
                      <input
                        type="text"
                        placeholder="Add tag..."
                        value={tagInput}
                        onChange={(e) => setTagInput(e.target.value)}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            addTag(tagInput);
                          }
                        }}
                        list="tag-suggestions"
                      />
                      <datalist id="tag-suggestions">
                        {availableTags.map(tag => (
                          <option key={tag} value={tag} />
                        ))}
                      </datalist>
                    </div>
                  </div>
                  
                  <div className="filter-group">
                    <label>Languages</label>
                    <div className="tags-input">
                      {preferences.languages.map(lang => (
                        <span key={lang} className="tag">
                          {lang}
                          <button onClick={() => removeLanguage(lang)}>×</button>
                        </span>
                      ))}
                      <select 
                        onChange={(e) => {
                          if (e.target.value) {
                            addLanguage(e.target.value);
                            e.target.value = '';
                          }
                        }}
                      >
                        <option value="">Select language...</option>
                        {availableLanguages.map(lang => (
                          <option key={lang} value={lang}>{lang}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  
                  <div className="filter-group">
                    <label>System Requirements</label>
                    <div className="system-specs">
                      <select
                        value={preferences.system_specs.os_type}
                        onChange={(e) => updateSystemSpec('os_type', e.target.value)}
                      >
                        <option value="">Any OS</option>
                        <option value="windows">Windows</option>
                        <option value="mac">Mac</option>
                        <option value="linux">Linux</option>
                      </select>
                      <input
                        type="number"
                        placeholder="RAM (GB)"
                        value={preferences.system_specs.memory_gb || ''}
                        onChange={(e) => updateSystemSpec('memory_gb', parseInt(e.target.value))}
                      />
                      <input
                        type="number"
                        placeholder="Storage (GB)"
                        value={preferences.system_specs.storage_gb || ''}
                        onChange={(e) => updateSystemSpec('storage_gb', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                </div>
                
                <button onClick={getRecommendations} className="btn-primary">
                  ✨ Get Personalized Recommendations
                </button>
              </div>
            )}
            
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Loading games...</p>
              </div>
            ) : (
              <div className="games-grid">
                {games.map((game, index) => (
                  <div 
                    key={index} 
                    className="game-card"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    {/* Game Poster */}
                    <div className="game-poster">
                      <img 
                        src={getGamePosterUrl(game) || getPlaceholderImage(game.title)}
                        alt={game.title}
                        className="poster-image"
                        onError={(e) => {
                          e.target.src = getPlaceholderImage(game.title);
                        }}
                      />
                      
                      {/* Discount badge */}
                      {game.discount_percentage > 0 && (
                        <div className="discount-badge">
                          -{Math.round(game.discount_percentage)}%
                        </div>
                      )}
                      
                      {/* Free badge */}
                      {game.discounted_price === 0 && (
                        <div className="free-badge">
                          FREE
                        </div>
                      )}
                    </div>
                    
                    <div className="card-glow"></div>
                    <div className="card-content">
                      <h3>{game.title}</h3>
                      
                      <div className="game-meta">
                        <span className="price">
                          {game.discounted_price === 0 ? 'FREE' : `$${game.discounted_price?.toFixed(2)}`}
                        </span>
                        {game.discount_percentage > 0 && (
                          <span className="discount">-{game.discount_percentage}%</span>
                        )}
                      </div>
                      
                      <div className="game-stats">
                        <div className="stat">
                          <span className="stat-icon">⭐</span>
                          <span>{(game.overall_sentiment_score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="stat">
                          <span className="stat-icon">👥</span>
                          <span>{game.all_reviews_count?.toLocaleString() || '0'}</span>
                        </div>
                      </div>
                      
                      {game.tags && (
                        <div className="game-tags">
                          {game.tags.slice(0, 3).map((tag, i) => (
                            <span key={i} className="tag">{tag}</span>
                          ))}
                        </div>
                      )}
                      
                      <div className="game-footer">
                        <span className="developer">{game.developer}</span>
                        {game.release_year && (
                          <span className="year">{game.release_year}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Recommendations Tab */}
        {activeTab === 'recommendations' && (
          <div className="tab-content fade-in">
            <div className="section-header">
              <h2>Personalized For You</h2>
              <p className="subtitle">Based on your preferences</p>
            </div>
            
            {recommendations.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">✨</div>
                <h3>No recommendations yet</h3>
                <p>Set your preferences and click "Get Recommendations" to discover games perfect for you!</p>
                <button onClick={() => setActiveTab('explore')} className="btn-primary">
                  Set Preferences
                </button>
              </div>
            ) : loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Finding perfect games for you...</p>
              </div>
            ) : (
              <div className="games-grid">
                {recommendations.map((rec, index) => (
                  <div 
                    key={index} 
                    className="game-card recommended"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    {/* Game Poster */}
                    <div className="game-poster">
                      <img 
                        src={getGamePosterUrl(rec) || getPlaceholderImage(rec.title)}
                        alt={rec.title}
                        className="poster-image"
                        onError={(e) => {
                          e.target.src = getPlaceholderImage(rec.title);
                        }}
                      />
                      
                      {/* Discount badge */}
                      {rec.discount > 0 && (
                        <div className="discount-badge">
                          -{Math.round(rec.discount)}%
                        </div>
                      )}
                      
                      {/* Free badge */}
                      {rec.price === 0 && (
                        <div className="free-badge">
                          FREE
                        </div>
                      )}
                    </div>
                    
                    <div className="card-glow"></div>
                    <div className="match-badge">
                      {(rec.score * 100).toFixed(0)}% Match
                    </div>
                    
                    <div className="card-content">
                      <h3>{rec.title}</h3>
                      
                      <div className="game-meta">
                        <span className="price">
                          {rec.price === 0 ? 'FREE' : `$${rec.price.toFixed(2)}`}
                        </span>
                        {rec.discount > 0 && (
                          <span className="discount">-{rec.discount}%</span>
                        )}
                      </div>
                      
                      {rec.explanations && rec.explanations.length > 0 && (
                        <div className="explanations">
                          {rec.explanations.map((exp, i) => (
                            <div key={i} className="explanation">
                              <span className="check">✓</span>
                              <span>{exp}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      <div className="game-stats">
                        <div className="stat">
                          <span className="stat-icon">⭐</span>
                          <span>{(rec.sentiment * 100).toFixed(0)}%</span>
                        </div>
                        <div className="stat">
                          <span className="stat-icon">🔥</span>
                          <span>{(rec.popularity * 100).toFixed(0)}%</span>
                        </div>
                        <div className="stat">
                          <span className="stat-icon">👥</span>
                          <span>{rec.reviews_count?.toLocaleString() || '0'}</span>
                        </div>
                      </div>
                      
                      {rec.tags && (
                        <div className="game-tags">
                          {rec.tags.slice(0, 4).map((tag, i) => (
                            <span key={i} className="tag">{tag}</span>
                          ))}
                        </div>
                      )}
                      
                      <div className="game-footer">
                        <span className="developer">{rec.developer}</span>
                        {rec.release_year && (
                          <span className="year">{rec.release_year}</span>
                        )}
                      </div>
                      
                      {rec.link && (
                        <a 
                          href={rec.link} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="view-link"
                        >
                          View on Steam →
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Stats Tab */}
        {activeTab === 'stats' && stats && (
          <div className="tab-content fade-in">
            <h2>Platform Statistics</h2>
            
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-icon">🎮</div>
                <div className="stat-value">{stats.total_games}</div>
                <div className="stat-label">Total Games</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">💰</div>
                <div className="stat-value">
                  ${stats.price_stats?.avg_price?.toFixed(2) || '0.00'}
                </div>
                <div className="stat-label">Average Price</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">🆓</div>
                <div className="stat-value">{stats.price_stats?.free_games || 0}</div>
                <div className="stat-label">Free Games</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-icon">💎</div>
                <div className="stat-value">
                  ${stats.price_stats?.max_price?.toFixed(2) || '0.00'}
                </div>
                <div className="stat-label">Highest Price</div>
              </div>
            </div>
            
            <div className="tags-cloud">
              <h3>Popular Tags</h3>
              <div className="cloud">
                {stats.top_tags?.slice(0, 20).map((tag, index) => (
                  <span 
                    key={index} 
                    className="cloud-tag"
                    style={{
                      fontSize: `${0.9 + (tag.count / stats.top_tags[0].count) * 0.8}em`
                    }}
                  >
                    {tag._id} ({tag.count})
                  </span>
                ))}
              </div>
            </div>
            
            <div className="languages-list">
              <h3>Supported Languages</h3>
              <div className="cloud">
                {stats.top_languages?.map((lang, index) => (
                  <span key={index} className="cloud-tag">
                    {lang._id} ({lang.count})
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>GameFinder • Knowledge-Based Recommendation System</p>
        <p className="tech-stack">FastAPI • React • MongoDB • Scikit-learn</p>
      </footer>
    </div>
  );
}

export default App;