import { useEffect, useRef, useState } from 'react';
import './ChatContainer.css';
import ReactMarkdown from 'react-markdown';

const EMOTION_COLORS = {
  joy: '#ffb6c1', love: '#ffb6c1', admiration: '#c084fc',
  excitement: '#fbbf24', gratitude: '#34d399', optimism: '#fbbf24',
  pride: '#fbbf24', amusement: '#fbbf24', approval: '#60a5fa',
  anger: '#f87171', annoyance: '#fb923c', disapproval: '#a78bfa',
  disgust: '#9ca3af', sadness: '#818cf8', grief: '#818cf8',
  disappointment: '#93c5fd', remorse: '#c4b5fd', fear: '#94a3b8',
  nervousness: '#6495ed', neutral: '#9db4c0', confusion: '#a3e635',
  curiosity: '#22d3ee', realization: '#2dd4bf', surprise: '#fb923c',
  caring: '#f472b6', desire: '#c084fc', embarrassment: '#fb923c',
  relief: '#4ade80',
};

const EMOTION_LABELS = {
  joy: 'Joy', love: 'Love', admiration: 'Admiration', 
  excitement: 'Excitement', gratitude: 'Gratitude', anger: 'Anger', 
  sadness: 'Sadness', fear: 'Fear', neutral: 'Neutral', confusion: 'Confusion'
};

// ─── SHAP EXPLAINABILITY COMPONENT ───
const ShapExplainability = ({ shapValues }) => {
  const [expanded, setExpanded] = useState(false);
  if (!shapValues || shapValues.length === 0) return null;

  return (
    <div style={{ width: '100%' }}>
      <button className="shap-toggle" onClick={() => setExpanded(!expanded)}>
        AI Analysis {expanded ? '▲' : '▼'}
      </button>
      
      {expanded && (
        <div style={{ marginTop: '0.5rem', background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: '12px', border: '1px solid var(--border-subtle)', width: '100%' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', fontWeight: 600 }}>Word Importance:</div>
          {shapValues.map((sv, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%', marginBottom: '6px' }}>
              <span style={{ width: '60px', fontSize: '0.8rem', color: 'var(--text-primary)', textAlign: 'right' }}>"{sv.word}"</span>
              <div style={{ flex: 1, height: '8px', background: 'var(--bg-primary)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(sv.impact * 200, 100)}%`, background: 'linear-gradient(90deg, #f59e0b, #ea580c)' }} />
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', width: '35px' }}>+{sv.impact.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── NEW: ANIMATED TYPING INDICATOR WITH AVATAR ───
const TypingIndicator = () => {
  const [phrase, setPhrase] = useState("Reflecting");

  useEffect(() => {
    const phrases = [
      "Reflecting", 
      "Analyzing your words...", 
      "Gathering my thoughts...", 
      "Thinking"
    ];
    let i = 0;
    const interval = setInterval(() => {
      i = (i + 1) % phrases.length;
      setPhrase(phrases[i]);
    }, 2500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="message-wrapper bot">
      {/* Container aligned as a row to put avatar next to bubble */}
      <div className="message-content-group" style={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-end', gap: '10px' }}>
        
        {/* The Glowing Avatar */}
        <div className="bot-avatar-thinking">
          <img src="/Logo-soulsync.png" alt="Soul-Sync" />
        </div>

        {/* The Typing Bubble */}
        <div className="message bot typing-bubble">
          <span className="typing-text">{phrase}</span>
          <div className="bouncing-dots">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        </div>

      </div>
    </div>
  );
};

// ─── MAIN CHAT CONTAINER ───
export const ChatContainer = ({ messages, isSending }) => {
  const endRef = useRef(null);

  // Auto-scroll runs when messages change OR when bot starts/stops sending
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isSending]);

  // ── EMPTY STATE (Welcome Message) ──
  // FIX: Removed the wrapping <div className="chat-container"> from both
  // return statements. The .chat-container already exists in App.jsx, so
  // nesting another one here was breaking the flex layout and causing the
  // input bar to scroll away instead of staying pinned at the bottom.
  if (!messages || messages.length === 0) {
    return (
      <div className="messages">
        <div className="message-wrapper bot">
           <div className="message-content-group">
              <div className="message bot">
                Welcome to Soul-Sync. I'm here to walk alongside you through
                whatever you're experiencing. This is a safe, judgment-free space.
              </div>
              <div className="bot-metadata">
                <div className="emotion-badge">
                  <span className="emotion-dot" style={{ background: '#9db4c0' }} />
                  <span className="emotion-label">Welcoming</span>
                </div>
              </div>
           </div>
        </div>
      </div>
    );
  }

  return (
    <div className="messages">
      {messages.map((msg, index) => (
        <div key={msg.id || `msg-${index}`} style={{ display: 'contents' }}>
          
          {/* USER MESSAGE (Aligns Right) */}
          {msg.type === 'user' && (
            <div className="message-wrapper user">
              <div className="message-content-group">
                <div className="message user">{msg.text}</div>
              </div>
            </div>
          )}

          {/* LEGACY TYPING INDICATOR (Replaced with new one just in case it triggers via array) */}
          {msg.type === 'typing' && (
            <TypingIndicator />
          )}

          {/* BOT MESSAGE (Aligns Left) */}
          {msg.type === 'bot' && (
            <div className="message-wrapper bot">
              <div className="message-content-group">
                <div className="message bot">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
                
                {/* Metadata sits cleanly underneath the left-aligned bubble */}
                <div className="bot-metadata">
                  {msg.crisis?.is_crisis && (
                    <div style={{ background: '#fef2f2', color: '#dc2626', border: '1px solid #f87171', padding: '4px 10px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 600, marginTop: '4px' }}>
                      ⚠️ {msg.crisis.severity} concern detected
                    </div>
                  )}
                  <ShapExplainability shapValues={msg.shap_values} />
                </div>
              </div>
            </div>
          )}

          {/* ERROR MESSAGE */}
          {msg.type === 'error' && (
            <div className="message-wrapper bot">
               <div className="message-content-group">
                  <div className="message" style={{ background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca' }}>
                    ⚠️ {msg.text}
                  </div>
               </div>
            </div>
          )}

        </div>
      ))}
      
      {/* NEW: Displays the animated avatar when the backend is processing */}
      {isSending && <TypingIndicator />}

      <div ref={endRef} />
    </div>
  );
};
