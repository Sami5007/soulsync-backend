import { useState } from 'react';
// ✅ 1. IMPORT THE FLAT FUNCTION INSTEAD OF 'api'
import { adminLogin } from '../services/api';
import './AdminLogin.css';

export const AdminLogin = ({ onLoginSuccess, onCancel }) => {
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [shake, setShake] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!password.trim() || loading) return;

    setError('');
    setLoading(true);

    try {
      // ✅ 2. CALL THE FLAT FUNCTION AND CHECK IF IT WORKED
      const success = await adminLogin(password);
      
      if (success) {
        onLoginSuccess();
      } else {
        throw new Error('Invalid password');
      }
    } catch (err) {
      setError(err.message || 'Invalid password');
      setShake(true);
      setPassword('');
      setTimeout(() => setShake(false), 600);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="admin-login-wrapper">
      <div className={`admin-login-card ${shake ? 'shake' : ''}`}>
        <div className="admin-login-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
          </svg>
        </div>

        <h1 className="admin-login-title">Admin Access</h1>
        <p className="admin-login-subtitle">Soul-Sync Analytics Dashboard</p>

        <form onSubmit={handleSubmit} className="admin-login-form">
          <div className="admin-password-wrapper">
            <input
              type={showPassword ? 'text' : 'password'}
              className="admin-password-input"
              placeholder="Enter admin password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoFocus
              disabled={loading}
            />
            <button
              type="button"
              className="admin-password-toggle"
              onClick={() => setShowPassword((v) => !v)}
              tabIndex={-1}
            >
              {showPassword ? '🙈' : '👁️'}
            </button>
          </div>

          {error && (
            <div className="admin-login-error">
              <span>⚠️</span> {error}
            </div>
          )}

          <button
            type="submit"
            className="admin-login-btn"
            disabled={!password.trim() || loading}
          >
            {loading ? (
              <span className="admin-spinner" />
            ) : (
              <>
                Sign In <span style={{ marginLeft: 8 }}>→</span>
              </>
            )}
          </button>
        </form>

        <button
          type="button"
          className="admin-login-back"
          onClick={onCancel}
          disabled={loading}
        >
          ← Back to Soul-Sync
        </button>

        <p className="admin-login-footer">
          🔒 Session is encrypted · Tokens expire in 4 hours
        </p>
      </div>
    </div>
  );
};
