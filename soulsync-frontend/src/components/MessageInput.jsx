import { useState, useRef, useEffect } from 'react';
import './MessageInput.css';

// ─── Check browser support ONCE at module load ───
const SpeechRecognition =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;
const VOICE_SUPPORTED = Boolean(SpeechRecognition);

export const MessageInput = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');
  const [isListening, setIsListening] = useState(false);
  const textareaRef = useRef(null);
  const recognitionRef = useRef(null);

  // Voice session buffers
  const prefixRef = useRef('');         // text typed before voice started
  const finalBufferRef = useRef('');    // accumulated finalized speech this session
  
  // 🎤 MOBILE PWA DUP FIX: Track unique phrase tokens to prevent mobile duplicates
  const mobileSeenPhrasesRef = useRef(new Set()); 
  
  // Flag: did the user manually stop, or did browser auto-stop?
  const userStoppedRef = useRef(false);

  const MAX_CHARS = 500;

  // ─── INITIALIZE SPEECH RECOGNITION ───
  useEffect(() => {
    if (!VOICE_SUPPORTED) return;

    const recognition = new SpeechRecognition();
    
    // MOBILE PWA FIX: continuous = true breaks native Android UI layers.
    // Changing this to false prevents the OS keyboard buffer from looping.
    recognition.continuous = false; 
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      mobileSeenPhrasesRef.current.clear(); // Fresh session, clear dup trackers
    };

    // 🎯 FIXED MOBILE EVENT HANDLING LOGIC
    recognition.onresult = (event) => {
      let finalText = finalBufferRef.current; // Build on top of what we already saved
      let interimText = '';

      // Loop only from the latest results index to prevent reading old Android array indexes
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const resultItem = event.results[i];
        const transcript = resultItem[0].transcript.trim();

        // MOBILE FIX: Filter out 0-confidence ghost streams sent by Android Chrome
        const isValidMobileFinal = resultItem.isFinal && resultItem[0].confidence > 0;

        if (isValidMobileFinal) {
          // MOBILE FIX: Deduplicate text matching tokens using our Set
          if (!mobileSeenPhrasesRef.current.has(transcript)) {
            mobileSeenPhrasesRef.current.add(transcript);
            finalText = finalText ? `${finalText} ${transcript}` : transcript;
          }
        } else if (!resultItem.isFinal) {
          interimText = transcript;
        }
      }

      // Keep finalBufferRef in sync
      finalBufferRef.current = finalText;

      // Compose string outputs safely
      const prefix = prefixRef.current;
      let combined = prefix;
      if (finalText) combined = combined ? `${combined} ${finalText}` : finalText;
      if (interimText) combined = combined ? `${combined} ${interimText}` : interimText;

      combined = combined.replace(/\s+/g, ' ').trim().slice(0, MAX_CHARS);
      setMessage(combined);

      // Auto-resize textarea
      if (textareaRef.current) {
        const el = textareaRef.current;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 120) + 'px';
      }
    };

    recognition.onerror = (event) => {
      console.error('[VoiceInput] Error:', event.error);
      if (event.error === 'no-speech' || event.error === 'aborted') return;

      if (event.error === 'not-allowed') {
        alert('Microphone permission denied. Please enable it in your browser settings.');
        userStoppedRef.current = true;
        setIsListening(false);
        return;
      }

      userStoppedRef.current = true;
      setIsListening(false);
    };

    // 🔄 MOBILE PWA AUTO-RESTART LOGIC
    recognition.onend = () => {
      // Clear token lookup index for the next sound check burst
      mobileSeenPhrasesRef.current.clear();

      if (!userStoppedRef.current) {
        try {
          const accumulated = [prefixRef.current, finalBufferRef.current]
            .filter(Boolean)
            .join(' ')
            .trim();
          prefixRef.current = accumulated;
          // Notice: Keep finalBufferRef populated so it persists across rapid restarts

          recognition.start();
          return;
        } catch (err) {
          console.warn('[VoiceInput] Auto-restart failed:', err);
        }
      }

      // Complete reset upon manual stop
      setIsListening(false);
      finalBufferRef.current = '';
      prefixRef.current = '';
      userStoppedRef.current = false;
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        userStoppedRef.current = true;
        try {
          recognitionRef.current.abort();
        } catch (err) {
          console.warn('[VoiceInput] Cleanup error:', err);
        }
      }
    };
  }, []);

  // ─── TOGGLE LISTENING ───
  const handleVoiceInput = () => {
    if (!VOICE_SUPPORTED || !recognitionRef.current) {
      alert("Your browser doesn't support voice input. Please use Chrome or Edge.");
      return;
    }

    if (isListening) {
      userStoppedRef.current = true;
      try {
        recognitionRef.current.stop();
      } catch (err) {
        console.warn('[VoiceInput] Stop error:', err);
      }
    } else {
      const current = message.trim();
      prefixRef.current = current;
      finalBufferRef.current = '';
      mobileSeenPhrasesRef.current.clear();
      userStoppedRef.current = false;

      try {
        recognitionRef.current.start();
      } catch (err) {
        console.error('[VoiceInput] Start error:', err);
        setIsListening(false);
      }
    }
  };

  // ─── SEND HANDLER ───
  const handleSubmit = async () => {
    if (!message.trim() || disabled || message.length > MAX_CHARS) return;
    const text = message.trim();
    setMessage('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    if (isListening && recognitionRef.current) {
      userStoppedRef.current = true;
      try {
        recognitionRef.current.stop();
      } catch (err) {
        console.warn('[VoiceInput] Stop error:', err);
      }
    }

    await onSendMessage(text);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChange = (e) => {
    if (e.target.value.length <= MAX_CHARS) {
      setMessage(e.target.value);
    }
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  };

  return (
    <div className="input-container">
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          className="message-input"
          placeholder={
            disabled
              ? 'Soul-Sync is reflecting... '
              : isListening
              ? '🎤 Listening... speak now (tap mic to stop)'
              : 'Type your message here...'
          }
          value={message}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
        />
        {message.length > 400 && (
          <div className={`char-counter ${message.length >= 450 ? 'warning' : ''}`}>
            {message.length} / {MAX_CHARS}
          </div>
        )}
      </div>

      {VOICE_SUPPORTED && (
        <button
          type="button"
          className={`voice-button ${isListening ? 'listening' : ''}`}
          onClick={handleVoiceInput}
          disabled={disabled}
          title={isListening ? 'Stop listening' : 'Voice input'}
          aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
          style={{
            backgroundColor: isListening ? 'rgba(239, 68, 68, 0.15)' : 'transparent',
            border: 'none',
            borderRadius: '50%',
            width: '38px',
            height: '38px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease',
            color: isListening ? '#ef4444' : '#94a3b8',
            marginRight: '8px',
            outline: 'none',
            boxShadow: 'none'
          }}
          onMouseEnter={(e) => {
            if (!isListening && !disabled) {
              e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.08)';
              e.currentTarget.style.color = '#ffffff';
            }
          }}
          onMouseLeave={(e) => {
            if (!isListening && !disabled) {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.color = '#94a3b8';
            }
          }}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
          </svg>
        </button>
      )}
    </div>
  );
};
