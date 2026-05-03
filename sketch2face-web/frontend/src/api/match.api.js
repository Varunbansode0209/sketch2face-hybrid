import api from './axios';

export const matchAPI = {
  // Run face matching (matches backend /api/match/run)
  run: async (file, gallery) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('gallery', gallery);
    
    const response = await api.post('/match/run', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get match history
  getHistory: async () => {
    const response = await api.get('/match/history');
    return response.data;
  },

  // Get specific match details
  getMatchById: async (queryId) => {
    const response = await api.get(`/match/${queryId}`);
    return response.data;
  },
};