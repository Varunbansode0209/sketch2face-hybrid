import api from './axios';

export const authAPI = {
  // Register new user
  register: async (userData) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },

  // Login user (backend uses OAuth2PasswordRequestForm - needs FormData)
  login: async (credentials) => {
    // Backend expects OAuth2PasswordRequestForm format
    const formData = new FormData();
    formData.append('username', credentials.email); // OAuth2 uses 'username' field
    formData.append('password', credentials.password);
    
    const response = await api.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get current user profile (backend endpoint is /auth/me)
  getProfile: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  // Logout
  logout: async () => {
    await api.post('/auth/logout');
  },
};