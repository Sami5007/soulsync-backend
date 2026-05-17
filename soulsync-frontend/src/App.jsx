import { useState, useEffect, useRef } from 'react';
import { ChatContainer } from './components/ChatContainer';
import { MessageInput } from './components/MessageInput';
import { sendCrisisEmail } from './services/crisisEmail';
import { CrisisAlert } from './components/CrisisAlert';
import { CrisisConsentModal } from './components/CrisisConsentModal';
import { PreferenceSelector } from './components/PreferenceSelector';
import { PreferenceModal } from './components/PreferenceModal';
import { AdminLogin } from './components/AdminLogin';
import { AdminDashboard } from './components/AdminDashboard';
import { api } from './services/api';
import './App.css';

/**
 * COMPONENT: Particles — emotion-aware colors
 * Reads the emotion prop and applies matching CSS class for particle color
 */
function Particles({ emotion }) {
  const containerRef = useRef(null);
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const count = 30;
    for (let i = 0; i < count; i++) {
      const p = document.createElement('div');
      p.className = 'particle';
      p.style.left = Math.random() * 100 + '%';
      p.style.animationDuration = (15 + Math.random() * 20) + 's';
      p.style.animationDelay = Math.random() * 10 + 's';
      container.appendChild(p);
    }
  }, []);
  // emotion class drives the particle color via CSS variable
  return <div className={`ambient-particles ${emotion ? `mood-particles-${emotion}` : ''}`} ref={containerRef} />;
}

function Navbar({ scrolled, onGetStarted }) {
  return (
    <nav className={`landing-nav ${scrolled ? 'nav-scrolled' : ''}`}>
      <div className="nav-logo">
        <img src="/Logo-soulsync.png" alt="Soul-Sync" className="nav-logo-img" />
        <span className="nav-logo-text">Soul-Sync</span>
      </div>
      <div className="nav-links">
        <a href="#features" className="nav-link">Features</a>
        <a href="#how-it-works" className="nav-link">How It Works</a>
        <a href="#safety" className="nav-link">Safety</a>
      </div>
      <button className="nav-cta" onClick={onGetStarted}>
        Get Started <span className="nav-cta-arrow">→</span>
      </button>
    </nav>
  );
}

function HeroSection({ onStartChat, onLearnMore }) {
  return (
    <section className="hero-section">
      <div className="hero-glow" />
      <div className="hero-content">
        <img src="/Logo-soulsync.png" alt="Soul-Sync Logo" className="hero-logo fade-up" />
        <div className="hero-badge fade-up delay-1">
          <span className="hero-badge-icon">✨</span>
          AI-Powered Mental Wellness Support
        </div>
        <h1 className="hero-title fade-up delay-2">
          Your compassionate companion for{' '}
          <span className="hero-title-accent">mental wellness</span>
        </h1>
        <p className="hero-desc fade-up delay-3">
          Soul-Sync understands your emotions and provides personalized support
          through evidence-based techniques or faith-based guidance. Start your
          journey to better mental health today.
        </p>
        <div className="hero-buttons fade-up delay-4">
          <button className="btn-primary" onClick={onStartChat}>
            Start Chatting <span className="btn-icon">💬</span>
          </button>
          <a href="#features" className="btn-secondary" onClick={onLearnMore}>
            Learn More
          </a>
        </div>
        <p className="hero-disclaimer fade-up delay-5">
          This system is not a replacement for professional mental healthcare.
          If you are in crisis, please contact emergency services.
        </p>
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    { icon: '🧠', iconClass: 'feature-icon-cyan', title: 'Emotion Detection', desc: 'Advanced AI understands your emotional state from your messages, providing responses tailored to how you feel.' },
    { icon: '✨', iconClass: 'feature-icon-cyan', title: 'Explainable AI', desc: 'Understand why specific emotions were detected with transparent SHAP explanations you can explore.' },
    { icon: '💬', iconClass: 'feature-icon-cyan', title: 'Personalized Guidance', desc: 'Choose between Islamic spiritual guidance or evidence-based psychological techniques based on your preference.' },
    { icon: '🛡️', iconClass: 'feature-icon-red', title: 'Crisis Detection', desc: 'Automatic detection of crisis situations with immediate access to emergency resources and professional help.' },
    { icon: '💭', iconClass: 'feature-icon-cyan', title: 'Natural Conversations', desc: 'Chat naturally in a safe, judgment-free space. Your conversation stays private and session-based.' },
    { icon: '📱', iconClass: 'feature-icon-cyan', title: 'Responsive Design', desc: 'Access Soul-Sync from any device. Optimized for both desktop and mobile experiences.' }
  ];
  return (
    <section id="features" className="features-section">
      <div className="section-header">
        <h2 className="section-title">Intelligent Support, Always Available</h2>
        <p className="section-subtitle">Soul-Sync combines advanced AI with evidence-based mental health practices to provide meaningful support whenever you need it.</p>
      </div>
      <div className="feature-grid">
        {features.map((f, i) => (
          <div key={i} className="feature-card">
            <div className={`feature-card-icon ${f.iconClass}`}>{f.icon}</div>
            <h3 className="feature-card-title">{f.title}</h3>
            <p className="feature-card-desc">{f.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function HowItWorksSection() {
  const steps = [
    { num: 1, title: 'Choose Your Mode', desc: 'Select Islamic Guidance for Quranic verses and spiritual support, or Evidence-Based Psychology for CBT and mindfulness techniques.' },
    { num: 2, title: 'Share How You Feel', desc: 'Type your thoughts and feelings naturally. Our AI detects your emotional state and provides appropriate support.' },
    { num: 3, title: 'Receive Guidance', desc: 'Get personalized responses, coping strategies, and resources tailored to your emotional needs and preferences.' }
  ];
  return (
    <section id="how-it-works" className="how-it-works-section">
      <div className="section-header">
        <h2 className="section-title">How Soul-Sync Works</h2>
        <p className="section-subtitle">Getting started is simple. Choose your preferred mode and begin your wellness journey.</p>
      </div>
      <div className="steps-row">
        {steps.map((s, i) => (
          <div key={i} className="step-item">
            <div className="step-number">{s.num}</div>
            <h3 className="step-title">{s.title}</h3>
            <p className="step-desc">{s.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function SafetySection() {
  return (
    <section id="safety" className="safety-section">
      <div className="safety-content">
        <div className="safety-badge"><span>🛡️</span> Your Safety Matters</div>
        <h2 className="safety-title">We're Here When You Need Help Most</h2>
        <p className="safety-desc">Soul-Sync automatically detects signs of crisis and provides immediate access to professional help. Your wellbeing is our priority.</p>
        <div className="emergency-card">
          <h3 className="emergency-heading">Emergency Resources</h3>
          <div className="emergency-numbers">
            <div className="emergency-item">
              <span className="emergency-label">Mental Health Helpline</span>
              <span className="emergency-number">1166</span>
            </div>
            <div className="emergency-item">
              <span className="emergency-label">Emergency Services</span>
              <span className="emergency-number">1122</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTASection({ onGetStarted }) {
  return (
    <section className="cta-section">
      <h2 className="cta-title">Ready to Start Your Wellness Journey?</h2>
      <p className="cta-desc">Join thousands who have found support through Soul-Sync. Begin your conversation today.</p>
      <button className="btn-primary cta-btn" onClick={onGetStarted}>
        Get Started Now <span className="btn-arrow">→</span>
      </button>
    </section>
  );
}

function Footer() {
  return (
    <footer className="landing-footer">
      <div className="footer-content">
        <div className="footer-logo">
          <img src="/Logo-soulsync.png" alt="Soul-Sync" className="footer-logo-img" />
          <span className="footer-logo-text">Soul-Sync</span>
        </div>
        <div className="footer-links">
          <a href="#" className="footer-link">Privacy Policy</a>
          <a href="#" className="footer-link">Terms of Service</a>
          <a href="#" className="footer-link">Crisis Resources</a>
          <a href="#admin" className="footer-link">Admin</a>
        </div>
        <span className="footer-copyright">© 2026 Soul-Sync. All rights reserved.</span>
      </div>
      <p className="footer-disclaimer">
        Disclaimer: Soul-Sync is not a substitute for professional mental health treatment.
        If you are experiencing a mental health emergency, please contact emergency services immediately.
      </p>
    </footer>
  );
}

function Sidebar({ open, sessions = [], activeSessionId, onNewChat, onSelectSession, isSending }) {
  return (
    <aside className={`sidebar ${open ? '' : 'collapsed'}`}>
      <button className="new-conversation-btn" onClick={onNewChat} disabled={isSending}>
        + New Chat
      </button>
      <div className="conversation-list">
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`conversation-item ${s.id === activeSessionId ? 'active' : ''}`}
            onClick={() => !isSending && onSelectSession(s.id)}
          >
            <div className="conversation-title">
              {s.messages && s.messages.length > 0
                ? s.messages[0].text.substring(0, 20) + "..."
                : "New Chat"}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}

/* ═══════════════════════════════════════════
   MAIN APP COMPONENT
   ═══════════════════════════════════════════ */
export default function App() {
  const [stage, setStage] = useState('welcome');
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [preference, setPreference] = useState('hybrid');
  const [loading, setLoading] = useState(false);
  const [crisisAlert, setCrisisAlert] = useState(null);
  const [appInfo, setAppInfo] = useState(null);
  const [isSending, setIsSending] = useState(false);
  const [theme, setTheme] = useState('dark');
  const [islamicMode, setIslamicMode] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [scrolled, setScrolled] = useState(false);

  // ── ADMIN STATE ──
  const [route, setRoute] = useState(window.location.hash);
  const [adminAuthed, setAdminAuthed] = useState(false);
  const [adminChecking, setAdminChecking] = useState(true);

  // ── CRISIS CONSENT STATE ──
  const [pendingCrisis, setPendingCrisis] = useState(null);

  // ── EMOTION-AWARE UI STATE ──
  const [currentEmotion, setCurrentEmotion] = useState(null);

  const activeSession = sessions.find(s => s.id === activeSessionId) || null;
  const messages = activeSession ? activeSession.messages : [];

  // ── HASH ROUTING ──
  useEffect(() => {
    const handleHashChange = () => setRoute(window.location.hash);
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    if (route === '#admin') {
      setAdminChecking(true);
      api.admin.verify().then(valid => {
        setAdminAuthed(valid);
        setAdminChecking(false);
      });
    }
  }, [route]);

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
    document.body.classList.toggle('islamic-mode', islamicMode);
  }, [theme, islamicMode]);

  useEffect(() => {
    const initApp = async () => {
      try {
        const info = await api.info();
        setAppInfo(info);
      } catch (e) {
        console.warn("API Info could not be fetched", e);
      }
    };
    initApp();
  }, []);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // ── API HANDLERS ──
  const startNewSession = async (pref) => {
    setLoading(true);
    // Reset emotion atmosphere on new session
    setCurrentEmotion(null);
    try {
      const response = await api.startSession(pref);
      const newSession = {
        id: response.session_id,
        preference: pref,
        date: new Date().toLocaleTimeString(),
        messages: []
      };
      setSessions(prev => [newSession, ...prev]);
      setActiveSessionId(response.session_id);
    } catch (e) {
      console.error('Failed to start session:', e);
      alert("Backend connection failed. Please ensure your Flask server is running.");
    } finally {
      setLoading(false);
    }
  };
 
  // ✅ FIXED: Now passes the actual conversation history to the backend
  // so the bot remembers prior turns instead of treating each message as new.
  const handleSendMessage = async (text) => {
    if (isSending || !activeSessionId) return;
    setIsSending(true);

    const userMsg = { id: Date.now(), type: 'user', text };

    // Build the up-to-date conversation list for THIS session
    // (using local var because setSessions is async and won't reflect immediately)
    const currentMessages = activeSession ? activeSession.messages : [];
    const updatedMessages = [...currentMessages, userMsg];

    // Optimistically render the user message
    setSessions(prev => prev.map(s =>
      s.id === activeSessionId ? { ...s, messages: updatedMessages } : s
    ));

    // Transform local message format → backend format
    // Backend expects: { sender: 'user' | 'bot', text: '...' }
    // Local format uses: { type: 'user' | 'bot', text: '...' }
    const historyForAPI = updatedMessages.map(m => ({
      sender: m.type === 'user' ? 'user' : 'bot',
      text: m.text
    }));

    try {
      // ✅ Pass historyForAPI as 4th arg so Grok sees prior context
      const response = await api.chat(text, activeSessionId, preference, historyForAPI);

      const botMsg = {
        id: Date.now() + 1,
        type: 'bot',
        text: response.response,
        emotion: response.emotion,
        shap_values: response.shap_values
      };

      setSessions(prev => prev.map(s =>
        s.id === activeSessionId ? { ...s, messages: [...s.messages, botMsg] } : s
      ));

      if (response.crisis && response.crisis.is_crisis) {
        sendCrisisEmail(
          text,
          response.crisis.severity,
          response.emotion,
          updatedMessages
        );
        setCrisisAlert(response.crisis);
      }
    } catch (e) {
      console.error("Chat error:", e);
    } finally {
      setIsSending(false);
    }
  };

  // ── CRISIS CONSENT HANDLERS ──
  const handleCrisisConsent = async () => {
    if (!pendingCrisis) return;

    const { userMessage, crisisData, emotion, history } = pendingCrisis;

    try {
      const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
        ? 'http://localhost:5000/api'
        : 'https://samikals-soulsyncai.hf.space/api';

      await fetch(`${API_BASE}/crisis/send-email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          severity: crisisData.severity,
          emotion: emotion,
          history: history.slice(-5).map(m => ({
            sender: m.type === 'user' ? 'user' : 'bot',
            text: m.text
          }))
        }),
      });
      console.log('[Crisis] Counselor email sent with user consent');
    } catch (err) {
      console.error('[Crisis] Failed to send email:', err);
    }

    setPendingCrisis(null);
  };

  const handleCrisisDecline = () => {
    console.log('[Crisis] User declined counselor notification');
    setPendingCrisis(null);
  };

  const handleDismissCrisis = () => {
    setCrisisAlert(null);
  };

  const handleStartJourney = () => setStage('preference');

  const handlePreferenceSelect = (selectedPref) => {
    setPreference(selectedPref);
    setIslamicMode(selectedPref === 'islamic');
    setStage('chat');
    if (sessions.length === 0) {
      startNewSession(selectedPref);
    }
  };

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  // ══════════════════════════════════════════
  // RENDER: ADMIN ROUTES
  // ══════════════════════════════════════════
  if (route === '#admin') {
    if (adminChecking) {
      return (
        <div style={{
          minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'radial-gradient(circle at 50% 30%, #1a2b4a 0%, #0a0f1e 100%)', color: '#f8fafc'
        }}>
          <div style={{
            width: 48, height: 48, border: '4px solid rgba(255,255,255,0.1)',
            borderTopColor: '#00d2ff', borderRadius: '50%',
            animation: 'adminSpin 0.8s linear infinite'
          }} />
        </div>
      );
    }

    if (!adminAuthed) {
      return (
        <AdminLogin
          onLoginSuccess={() => setAdminAuthed(true)}
          onCancel={() => { window.location.hash = ''; }}
        />
      );
    }

    return (
      <AdminDashboard
        onLogout={() => { setAdminAuthed(false); window.location.hash = ''; }}
      />
    );
  }

  // ══════════════════════════════════════════
  //  RENDER: WELCOME
  // ══════════════════════════════════════════
  if (stage === 'welcome') {
    return (
      <div className="landing-wrapper">
        <Particles />
        <Navbar scrolled={scrolled} onGetStarted={handleStartJourney} />
        <HeroSection onStartChat={handleStartJourney} />
        <FeaturesSection />
        <HowItWorksSection />
        <SafetySection />
        <CTASection onGetStarted={handleStartJourney} />
        <Footer />
      </div>
    );
  }

  // ══════════════════════════════════════════
  //  RENDER: PREFERENCE
  // ══════════════════════════════════════════
  if (stage === 'preference') {
    return (
      <div className="preference-overlay">
        <Particles />
        <PreferenceModal
          onConfirm={handlePreferenceSelect}
          onCancel={() => setStage('welcome')}
        />
      </div>
    );
  }

  // ══════════════════════════════════════════
  //  RENDER: CHAT INTERFACE — with emotion-aware atmosphere
  // ══════════════════════════════════════════

  // Build the emotion CSS class for the wrapper
  const emotionClass = currentEmotion ? `mood-${currentEmotion}` : '';

  return (
    <>
      {/* ─── EMOTION ATMOSPHERE LAYER (behind everything) ─── */}
      <div className={`emotion-atmosphere ${emotionClass}`} />

      {/* ─── EMOTION GLOW ORB (floating accent light) ─── */}
      <div className={`emotion-glow-orb ${emotionClass}`} />

      <Particles emotion={currentEmotion} />

      <div className={`app ${emotionClass}`}>
        <header className="header">
          <div className="logo-container">
            <img src="/Logo-soulsync.png" alt="Soul-Sync" className="logo-image" />
            <div className="logo-text">
              <div className="logo-title">Soul-Sync</div>
              <div className="logo-tagline">Your Wellness Companion</div>
            </div>
          </div>
          <div className="header-controls">
            {/* ─── EMOTION INDICATOR (shows current mood as a small badge) ─── */}
{currentEmotion && (
  <div className={`emotion-indicator ${emotionClass}`}>
    <span className="emotion-indicator-emoji">
      {{ joy: '😄', sadness: '😢', anger: '😠', fear: '😨', confusion: '😕', neutral: '😐' }[currentEmotion] || '🧠'}
    </span>
    <span className="emotion-indicator-label">{currentEmotion}</span>
  </div>
)}
            <div className="control-button" onClick={() => setSidebarOpen(!sidebarOpen)}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
            </div>

            <PreferenceSelector preference={preference} onPreferenceChange={setPreference} />
          </div>
        </header>

        <div className="main-content">
          <Sidebar
            open={sidebarOpen}
            sessions={sessions}
            activeSessionId={activeSessionId}
            onNewChat={() => startNewSession(preference)}
            onSelectSession={setActiveSessionId}
            isSending={isSending}
          />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <ChatContainer messages={messages} isSending={isSending} />
  <div className={`chat-input-area ${emotionClass}`}>
    <MessageInput onSendMessage={handleSendMessage} disabled={!activeSessionId || isSending} />
  </div>
</div>
        </div>

        {crisisAlert && (
          <CrisisAlert
            isCrisis={crisisAlert.is_crisis}
            crisisData={crisisAlert}
            onDismiss={handleDismissCrisis}
          />
        )}

        {pendingCrisis && (
          <CrisisConsentModal
            crisisData={pendingCrisis.crisisData}
            onConsent={handleCrisisConsent}
            onDecline={handleCrisisDecline}
          />
        )}
      </div>
    </>
  );
}
