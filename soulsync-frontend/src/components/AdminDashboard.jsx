import { useState, useEffect, useCallback } from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, LineChart, Line, Legend
} from 'recharts';
// ✅ IMPORT THE FLAT FUNCTIONS TO BYPASS MINIFICATION ERRORS
import { getAdminStats, adminLogout } from '../services/api';
import './AdminDashboard.css';

// Color palette matching your glassmorphism theme
const EMOTION_COLORS = {
  joy: '#fbbf24',
  sadness: '#818cf8',
  anger: '#f87171',
  fear: '#94a3b8',
  neutral: '#64748b',
  confusion: '#a3e635',
  disgust: '#9ca3af',
  surprise: '#fb923c',
};

const SEVERITY_COLORS = {
  critical: '#dc2626',
  high: '#f59e0b',
  medium: '#facc15',
};

const PREF_COLORS = {
  islamic: '#10b981',
  psychological: '#3a7bd5',
  hybrid: '#00d2ff',
};

export const AdminDashboard = ({ onLogout }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadStats = useCallback(async (showLoader = false) => {
    if (showLoader) setLoading(true);
    try {
      // ✅ CALL THE FLAT FUNCTION HERE
      const data = await getAdminStats();
      setStats(data);
      setLastUpdated(new Date());
      setError('');
    } catch (err) {
      setError(err.message || 'Failed to load stats');
      if (err.message?.includes('expired') || err.message?.includes('authenticated')) {
        setTimeout(() => onLogout(), 1500);
      }
    } finally {
      setLoading(false);
    }
  }, [onLogout]);

  // Initial load + auto-refresh every 30s
  useEffect(() => {
    loadStats(true);
    const interval = setInterval(() => loadStats(false), 30000);
    return () => clearInterval(interval);
  }, [loadStats]);

  const handleLogout = async () => {
    // ✅ CALL THE FLAT FUNCTION HERE
    adminLogout();
    onLogout();
  };

  if (loading) {
    return (
      <div className="admin-loading">
        <div className="admin-spinner-large" />
        <p>Loading analytics...</p>
      </div>
    );
  }

  if (error && !stats) {
    return (
      <div className="admin-error-screen">
        <div>⚠️</div>
        <p>{error}</p>
        <button onClick={() => loadStats(true)}>Retry</button>
      </div>
    );
  }

  const { overview = {}, emotion_distribution = [], crisis_breakdown = [],
          hourly_activity = [], preference_breakdown = [], recent_crises = [],
          daily_activity = [] } = stats || {};

  // Format data for charts
  const emotionChartData = emotion_distribution.map(e => ({
    name: e.emotion,
    value: e.count,
    color: EMOTION_COLORS[e.emotion] || '#64748b',
  }));

  const prefChartData = preference_breakdown.map(p => ({
    name: p.preference,
    value: p.count,
    color: PREF_COLORS[p.preference] || '#64748b',
  }));

  const totalPrefs = prefChartData.reduce((s, p) => s + p.value, 0) || 1;

  const totalEmotion = emotionChartData.reduce((s, e) => s + e.value, 0) || 1;

  return (
    <div className="admin-dashboard">
      {/* ─── TOP BAR ─── */}
      <header className="admin-header">
        <div className="admin-header-left">
          <div className="admin-logo">
            <span className="admin-logo-icon">🛡️</span>
            <div>
              <h1>Soul-Sync Admin</h1>
              <p>Analytics Dashboard</p>
            </div>
          </div>
        </div>
        <div className="admin-header-right">
          {lastUpdated && (
            <span className="admin-last-updated">
              <span className="admin-pulse-dot" />
              Updated {formatTime(lastUpdated)}
            </span>
          )}
          <button className="admin-refresh-btn" onClick={() => loadStats(true)} title="Refresh">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10"></polyline>
              <polyline points="1 20 1 14 7 14"></polyline>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
            </svg>
          </button>
          <button className="admin-logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </header>

      <div className="admin-body">
        {/* ─── OVERVIEW STAT CARDS ─── */}
        <div className="admin-stat-grid">
          <StatCard
            icon="💬"
            label="Total Messages"
            value={overview.total_messages || 0}
            accent="#00d2ff"
          />
          <StatCard
            icon="👥"
            label="Sessions"
            value={overview.total_sessions || 0}
            accent="#3a7bd5"
          />
          <StatCard
            icon="🚨"
            label="Crises Detected"
            value={overview.total_crises || 0}
            accent="#f87171"
            critical={overview.total_crises > 0}
          />
          <StatCard
            icon="📊"
            label="Messages Today"
            value={overview.messages_today || 0}
            accent="#10b981"
          />
          <StatCard
            icon="⚡"
            label="Avg Response"
            value={`${overview.avg_response_ms || 0}ms`}
            accent="#a855f7"
          />
        </div>

        {/* ─── CHARTS ROW 1: EMOTION + CRISIS ─── */}
        <div className="admin-grid-2col">
          <DashboardCard title="🎭 Emotion Distribution" subtitle="What users feel most">
            {emotionChartData.length === 0 ? (
              <EmptyState message="No emotion data yet" />
            ) : (
              <>
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie
                      data={emotionChartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={90}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {emotionChartData.map((e, i) => (
                        <Cell key={i} fill={e.color} stroke="rgba(255,255,255,0.1)" />
                      ))}
                    </Pie>
                    <Tooltip content={<DarkTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="admin-legend">
                  {emotionChartData.map((e, i) => (
                    <div key={i} className="admin-legend-item">
                      <span className="admin-legend-dot" style={{ background: e.color }} />
                      <span className="admin-legend-name">{e.name}</span>
                      <span className="admin-legend-value">
                        {Math.round((e.value / totalEmotion) * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </DashboardCard>

          <DashboardCard title="🚨 Crisis Severity" subtitle="Breakdown by tier">
            {crisis_breakdown.length === 0 ? (
              <EmptyState message="No crisis events — that's a good thing" emoji="✨" />
            ) : (
              <div className="admin-crisis-list">
                {crisis_breakdown.map((c, i) => {
                  const total = crisis_breakdown.reduce((s, x) => s + x.count, 0);
                  const pct = Math.round((c.count / total) * 100);
                  return (
                    <div key={i} className="admin-crisis-row">
                      <div className="admin-crisis-header">
                        <span
                          className="admin-crisis-badge"
                          style={{ background: SEVERITY_COLORS[c.severity] }}
                        >
                          {c.severity?.toUpperCase()}
                        </span>
                        <span className="admin-crisis-count">{c.count}</span>
                      </div>
                      <div className="admin-crisis-bar-bg">
                        <div
                          className="admin-crisis-bar-fill"
                          style={{
                            width: `${pct}%`,
                            background: SEVERITY_COLORS[c.severity],
                          }}
                        />
                      </div>
                      <span className="admin-crisis-pct">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            )}
          </DashboardCard>
        </div>

        {/* ─── CHART 2: HOURLY ACTIVITY ─── */}
        <DashboardCard title="⏰ Activity by Hour (last 7 days)" subtitle="When users seek support most">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={hourly_activity}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="hour"
                tickFormatter={(h) => `${h}:00`}
                stroke="rgba(255,255,255,0.4)"
                style={{ fontSize: '0.75rem' }}
              />
              <YAxis
                stroke="rgba(255,255,255,0.4)"
                style={{ fontSize: '0.75rem' }}
                allowDecimals={false}
              />
              <Tooltip content={<DarkTooltip suffix=" messages" />} />
              <Bar dataKey="count" fill="url(#hourGradient)" radius={[6, 6, 0, 0]} />
              <defs>
                <linearGradient id="hourGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00d2ff" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#3a7bd5" stopOpacity={0.6} />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </DashboardCard>

        {/* ─── ROW 3: DAILY TREND + PREFERENCE ─── */}
        <div className="admin-grid-2col">
          <DashboardCard title="📈 Daily Trend" subtitle="Last 7 days">
            {daily_activity.length === 0 ? (
              <EmptyState message="No daily data yet" />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={daily_activity}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="day"
                    stroke="rgba(255,255,255,0.4)"
                    style={{ fontSize: '0.7rem' }}
                    tickFormatter={(d) => d?.slice(5)}
                  />
                  <YAxis stroke="rgba(255,255,255,0.4)" style={{ fontSize: '0.75rem' }} allowDecimals={false} />
                  <Tooltip content={<DarkTooltip suffix=" messages" />} />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="#00d2ff"
                    strokeWidth={3}
                    dot={{ fill: '#00d2ff', r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </DashboardCard>

          <DashboardCard title="🎨 Preference Modes" subtitle="Islamic vs Psychological vs Hybrid">
            {prefChartData.length === 0 ? (
              <EmptyState message="No preference data yet" />
            ) : (
              <div className="admin-pref-bars">
                {prefChartData.map((p, i) => {
                  const pct = Math.round((p.value / totalPrefs) * 100);
                  return (
                    <div key={i} className="admin-pref-row">
                      <div className="admin-pref-label">
                        <span>{p.name}</span>
                        <strong>{p.value}</strong>
                      </div>
                      <div className="admin-pref-bar-bg">
                        <div
                          className="admin-pref-bar-fill"
                          style={{ width: `${pct}%`, background: p.color }}
                        />
                      </div>
                      <div className="admin-pref-pct">{pct}%</div>
                    </div>
                  );
                })}
              </div>
            )}
          </DashboardCard>
        </div>

        {/* ─── RECENT CRISES TABLE ─── */}
        <DashboardCard
          title="📋 Recent Crisis Alerts"
          subtitle="Metadata only — no message content stored (privacy by design)"
        >
          {recent_crises.length === 0 ? (
            <EmptyState message="No recent crises" emoji="✨" />
          ) : (
            <div className="admin-table-wrapper">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Session</th>
                    <th>Emotion</th>
                    <th>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {recent_crises.map((c, i) => (
                    <tr key={i}>
                      <td>{formatRelativeTime(c.timestamp)}</td>
                      <td><code>***{c.session}</code></td>
                      <td>
                        <span className="admin-emotion-pill" style={{
                          background: EMOTION_COLORS[c.emotion] + '33',
                          color: EMOTION_COLORS[c.emotion],
                          border: `1px solid ${EMOTION_COLORS[c.emotion]}55`
                        }}>
                          {c.emotion}
                        </span>
                      </td>
                      <td>
                        <span className="admin-severity-pill" style={{
                          background: SEVERITY_COLORS[c.severity] + '33',
                          color: SEVERITY_COLORS[c.severity],
                          border: `1px solid ${SEVERITY_COLORS[c.severity]}55`
                        }}>
                          {c.severity?.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </DashboardCard>

        <p className="admin-footer">
          🔒 All data is anonymized. No message content is stored in the analytics database.
          <br />
          Auto-refreshes every 30 seconds.
        </p>
      </div>
    </div>
  );
};

// ─── HELPER COMPONENTS ───
const StatCard = ({ icon, label, value, accent, critical }) => (
  <div className={`admin-stat-card ${critical ? 'critical' : ''}`}>
    <div className="admin-stat-icon" style={{ background: `${accent}22` }}>
      <span>{icon}</span>
    </div>
    <div className="admin-stat-body">
      <div className="admin-stat-value" style={{ color: accent }}>{value}</div>
      <div className="admin-stat-label">{label}</div>
    </div>
  </div>
);

const DashboardCard = ({ title, subtitle, children }) => (
  <div className="admin-card">
    <div className="admin-card-header">
      <h2>{title}</h2>
      {subtitle && <p>{subtitle}</p>}
    </div>
    <div className="admin-card-body">{children}</div>
  </div>
);

const EmptyState = ({ message, emoji = '📊' }) => (
  <div className="admin-empty">
    <div>{emoji}</div>
    <p>{message}</p>
  </div>
);

const DarkTooltip = ({ active, payload, label, suffix = '' }) => {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="admin-tooltip">
      {label !== undefined && <div className="admin-tooltip-label">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="admin-tooltip-row">
          <span>{p.name || 'Count'}:</span>
          <strong>{p.value}{suffix}</strong>
        </div>
      ))}
    </div>
  );
};

// ─── UTILITIES ───
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatRelativeTime(timestamp) {
  if (!timestamp) return '—';
  const date = new Date(timestamp);
  const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}
