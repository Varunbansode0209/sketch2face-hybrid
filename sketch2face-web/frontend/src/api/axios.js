import axios from 'axios';
import { getToken, removeToken } from '../utils/token';

export const API_BASE_URL = (() => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (envUrl) {
    return envUrl.endsWith('/api') ? envUrl.slice(0, -4) : envUrl;
  }
  // If hosted on Vercel or any non-localhost provider, gracefully default to the cloud backend
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
    return 'https://varunb2-sketch2face-api.hf.space';
  }
  return 'http://localhost:8000';
})();

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      removeToken();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;