import React from 'react';
import './GameDetailModal.css';

const GameDetailModal = ({ game, onClose, onOpenSteam }) => {
  if (!game) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        
        <div className="modal-header">
          <h2 className="modal-title">{game.title}</h2>
          <div className="modal-subtitle">
            <span>{game.developer}</span>
            {game.publisher && <span> • Published by {game.publisher}</span>}
            {game.release_year && <span> • Released {game.release_year}</span>}
          </div>
        </div>

        <div className="modal-body">
          <div className="game-main-info">
            <div className="game-price-section">
              <div className="price-display">
                <span className="current-price">
                  {game.price === 0 ? 'FREE' : `$${game.price.toFixed(2)}`}
                </span>
                {game.original_price > game.price && (
                  <>
                    <span className="original-price">${game.original_price.toFixed(2)}</span>
                    <span className="discount-badge">-{game.discount}%</span>
                  </>
                )}
              </div>
              
              {game.link && (
                <button 
                  className="steam-button"
                  onClick={() => onOpenSteam(game.link)}
                >
                  <span>🎮 View on Steam</span>
                </button>
              )}
            </div>

            <div className="game-stats-grid">
              <div className="stat-item">
                <div className="stat-label">Overall Rating</div>
                <div className="stat-value">{Math.round(game.sentiment * 100)}%</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Total Reviews</div>
                <div className="stat-value">{game.reviews_count?.toLocaleString() || '0'}</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Recent Reviews</div>
                <div className="stat-value">{game.recent_reviews?.toLocaleString() || '0'}</div>
              </div>
              <div className="stat-item">
                <div className="stat-label">Popularity Score</div>
                <div className="stat-value">{Math.round(game.popularity_score * 100)}%</div>
              </div>
            </div>
          </div>

          <div className="game-description">
            <h3>Description</h3>
            <p>{game.description || 'No description available.'}</p>
          </div>

          {game.tags && game.tags.length > 0 && (
            <div className="game-tags-section">
              <h3>Tags</h3>
              <div className="tags-cloud">
                {game.tags.map((tag, index) => (
                  <span key={index} className="tag-item">{tag}</span>
                ))}
              </div>
            </div>
          )}

          {game.features && game.features.length > 0 && (
            <div className="game-features">
              <h3>Features</h3>
              <div className="features-list">
                {game.features.map((feature, index) => (
                  <div key={index} className="feature-item">
                    <span className="feature-check">✓</span>
                    <span>{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {game.languages && game.languages.length > 0 && (
            <div className="game-languages">
              <h3>Supported Languages</h3>
              <div className="languages-grid">
                {game.languages.map((lang, index) => (
                  <span key={index} className="language-item">{lang}</span>
                ))}
              </div>
            </div>
          )}

          {game.minimum_requirements && (
            <div className="system-requirements">
              <h3>System Requirements</h3>
              <div className="requirements-content">
                <div className="requirements-grid">
                  {game.os_type && (
                    <div className="requirement-item">
                      <div className="req-label">OS</div>
                      <div className="req-value">{game.os_type}</div>
                    </div>
                  )}
                  {game.memory_gb && (
                    <div className="requirement-item">
                      <div className="req-label">Memory</div>
                      <div className="req-value">{game.memory_gb} GB RAM</div>
                    </div>
                  )}
                  {game.storage_gb && (
                    <div className="requirement-item">
                      <div className="req-label">Storage</div>
                      <div className="req-value">{game.storage_gb} GB available space</div>
                    </div>
                  )}
                  {game.gpu_brand && (
                    <div className="requirement-item">
                      <div className="req-label">Graphics</div>
                      <div className="req-value">{game.gpu_brand}</div>
                    </div>
                  )}
                  {game.cpu_brand && (
                    <div className="requirement-item">
                      <div className="req-label">Processor</div>
                      <div className="req-value">{game.cpu_brand}</div>
                    </div>
                  )}
                </div>
                <div className="requirements-text">
                  <pre>{game.minimum_requirements}</pre>
                </div>
              </div>
            </div>
          )}

          {game.explanations && game.explanations.length > 0 && (
            <div className="recommendation-reasons">
              <h3>Why We Recommend This Game</h3>
              <div className="reasons-list">
                {game.explanations.map((reason, index) => (
                  <div key={index} className="reason-item">
                    <span className="reason-icon">✨</span>
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GameDetailModal;