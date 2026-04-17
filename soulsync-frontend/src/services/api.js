// API Service for SOUL-SYNC Backend
// If testing on your laptop, use localhost. If on Vercel, use Hugging Face!
const API_BASE_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? 'http://localhost:5000/api'
  : 'https://samikals-soulsyncai.hf.space/api';
export const api = {
  // Health check
  health: async () => {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Backend not responding');
    return response.json();
  },

  // Get app info
  info: async () => {
    const response = await fetch(`${API_BASE_URL}/info`);
    if (!response.ok) throw new Error('Could not fetch app info');
    return response.json();
  },

  // Start a new session
  startSession: async (preference = 'hybrid') => {
    const response = await fetch(`${API_BASE_URL}/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preference }),
    });
    if (!response.ok) throw new Error('Could not start session');
    return response.json();
  },

  // Get session info
  getSession: async (sessionId) => {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);
    if (!response.ok) throw new Error('Session not found');
    return response.json();
  },

  // Send chat message — returns raw backend response, no transformation
  chat: async (message, sessionId, preference = 'hybrid') => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        preference,
      }),
    });
    if (!response.ok) throw new Error('Chat request failed');
    // ✅ Return raw data — no transformation
    // Backend returns: { emotion, confidence, response, crisis, ... }
    return response.json();
  },

  // Detect emotion only
  detectEmotion: async (message) => {
    const response = await fetch(`${API_BASE_URL}/emotion/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    if (!response.ok) throw new Error('Emotion detection failed');
    return response.json();
  },

  // Get crisis resources
  getCrisisResources: async () => {
    const response = await fetch(`${API_BASE_URL}/crisis/resources`);
    if (!response.ok) throw new Error('Could not fetch crisis resources');
    return response.json();
  },
};