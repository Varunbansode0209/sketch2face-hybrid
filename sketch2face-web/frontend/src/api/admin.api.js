import api from './axios';

export const adminAPI = {
  // Upload suspect to gallery (matches backend /api/admin/upload-suspect)
  uploadFace: async (formData) => {
    // formData is constructed in Admin.jsx directly and passed here.
    const response = await api.post('/admin/upload-suspect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Rebuild gallery embeddings (matches backend /api/admin/rebuild-gallery)
  rebuildGallery: async (gallery) => {
    const response = await api.post('/admin/rebuild-gallery', null, {
      params: { gallery },
    });
    return response.data;
  },

  // Get system logs (matches backend /api/admin/logs)
  getLogs: async (limit = 50) => {
    const response = await api.get('/admin/logs', {
      params: { limit },
    });
    return response.data;
  },

  // Get statistics (matches backend /api/admin/statistics)
  getStatistics: async () => {
    const response = await api.get('/admin/statistics');
    return response.data;
  },

  // List suspects (matches backend /api/admin/suspects)
  listSuspects: async (gallery = null, limit = 50) => {
    const response = await api.get('/admin/suspects', {
      params: { gallery, limit },
    });
    return response.data;
  },

  getAllUsers: async () => {
    const response = await api.get('/admin/users');
    return response.data;
  },

  deleteUser: async (userId) => {
    const response = await api.delete(`/admin/users/${userId}`);
    return response.data;
  },

  getAllFaces: async (gallery = 'all') => {
    const params = gallery && gallery !== 'all' ? { gallery } : {};
    const response = await api.get('/admin/suspects', { params });
    return { faces: response.data }; // Match expected structure in Admin.jsx
  },

  uploadFace: async (formData) => {
    const response = await api.post('/admin/upload-suspect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  deleteFace: async (suspectId) => {
    const response = await api.delete(`/admin/suspects/${suspectId}`);
    return response.data;
  }
};