import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:8000',
});

export const gameAPI = {
  // Health check
  healthCheck: () => API.get('/health'),
  
  // Get paginated games
  getGames: (page = 1, limit = 20, sortBy = 'popularity_score', sortOrder = -1) => 
    API.get(`/games?page=${page}&limit=${limit}&sort_by=${sortBy}&sort_order=${sortOrder}`),
  
  // Search games
  searchGames: (query) => 
    API.get(`/games/search?q=${encodeURIComponent(query)}`),
  
  // Filter games
  filterGames: (filters) => 
    API.post('/games/filter', filters),
  
  // Get recommendations based on preferences
  getRecommendations: (preferences) => 
    API.post('/recommend/preferences', {
      preferences: preferences,
      limit: 10
    }),
  
  // Get similar games
  getSimilarGames: (gameTitle, limit = 5) => 
    API.get(`/recommend/similar/${encodeURIComponent(gameTitle)}?limit=${limit}`),
  
  // Get stats
  getStats: () => API.get('/stats'),
  
  // Get tags
  getTags: () => API.get('/tags'),
  
  // Get categories
  getCategories: () => API.get('/categories'),
};