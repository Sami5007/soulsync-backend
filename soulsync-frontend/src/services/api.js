// api.js // API Service for SOUL-SYNC Backend
const API_BASE_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? 'http://localhost:5000/api'
  : 'https://samikals-soulsyncai.hf.space/api';

// ─── Admin token helpers ───
const ADMIN_TOKEN_KEY = 'soulsync_admin_token';
export const getAdminToken = () => sessionStorage.getItem(ADMIN_TOKEN_KEY);
export const setAdminToken = (token) => sessionStorage.setItem(ADMIN_TOKEN_KEY, token);
export const clearAdminToken = () => sessionStorage.removeItem(ADMIN_TOKEN_KEY);

export const api = {
  health: async () => {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Backend not responding');
    return response.json();
  },
  info: async () => {
    const response = await fetch(`${API_BASE_URL}/info`);
    if (!response.ok) throw new Error('Could not fetch app info');
    return response.json();
  },
  startSession: async (preference = 'hybrid') => {
    const response = await fetch(`${API_BASE_URL}/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preference }),
    });
    if (!response.ok) throw new Error('Could not start session');
    return response.json();
  },
  getSession: async (sessionId) => {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);
    if (!response.ok) throw new Error('Session not found');
    return response.json();
  },
  chat: async (message, sessionId, preference = 'hybrid', history = []) => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId, preference, history }),
    });
    if (!response.ok) throw new Error('Chat request failed');
    return response.json();
  },
  detectEmotion: async (message) => {
    const response = await fetch(`${API_BASE_URL}/emotion/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    if (!response.ok) throw new Error('Emotion detection failed');
    return response.json();
  },
  getCrisisResources: async () => {
    const response = await fetch(`${API_BASE_URL}/crisis/resources`);
    if (!response.ok) throw new Error('Could not fetch crisis resources');
    return response.json();
  }
};

// ─── ADMIN FLATTENED FUNCTIONS ───
export const adminVerify = async () => {
  try {
    const token = getAdminToken();
    if (!token) return false;
    const response = await fetch(`${API_BASE_URL}/admin/verify`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await response.json();
    return data.valid === true;
  } catch {
    return false;
  }
};

export const adminLogin = async (password) => {
  const response = await fetch(`${API_BASE_URL}/admin/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password })
  });
  const data = await response.json();
  if (data.success && data.token) {
    setAdminToken(data.token);
    return true;
  }
  return false;
};

export const adminLogout = () => {
  clearAdminToken();
};

export const getAdminStats = async () => {
  const token = getAdminToken();
  const response = await fetch(`${API_BASE_URL}/admin/analytics`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  if (response.status === 401) {
      clearAdminToken();
      window.location.reload(); 
  }
  if (!response.ok) throw new Error('Failed to fetch analytics');
  return response.json();
};
