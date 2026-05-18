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
  // Flag: did the user manually stop, or did browser auto-stop?
  const userStoppedRef = useRef(false);

  const MAX_CHARS = 500;

  // ─── INITIALIZE SPEECH RECOGNITION ───
  useEffect(() => {
    if (!VOICE_SUPPORTED) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;        // ✅ keep listening across pauses
    recognition.interimResults = true;    // show partials while speaking
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsListening(true);

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let newFinalText = '';

      // Iterate only NEW results (from resultIndex onward)
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          newFinalText += ' ' + transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      // Append only the NEW finalized text to the buffer
      if (newFinalText.trim()) {
        finalBufferRef.current = (
          finalBufferRef.current + ' ' + newFinalText
        )
          .replace(/\s+/g, ' ')
          .trim();
      }

      // Compose: original prefix + finalized voice + current interim
      const prefix = prefixRef.current;
      const finalVoice = finalBufferRef.current;
      const interim = interimTranscript.trim();

      let combined = prefix;
      if (finalVoice) {
        combined = combined ? `${combined} ${finalVoice}` : finalVoice;
      }
      if (interim) {
        combined = combined ? `${combined} ${interim}` : interim;
      }

      combined = combined.slice(0, MAX_CHARS);
      setMessage(combined);

      // Auto-resize
      if (textareaRef.current) {
        const el = textareaRef.current;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 120) + 'px';
      }
    };

    recognition.onerror = (event) => {
      console.error('[VoiceInput] Error:', event.error);

      // 'no-speech' and 'aborted' are normal — don't kill the session
      if (event.error === 'no-speech' || event.error === 'aborted') {
        return;
      }

      if (event.error === 'not-allowed') {
        alert('Microphone permission denied. Please enable it in your browser settings.');
        userStoppedRef.current = true;
        setIsListening(false);
        return;
      }

      // Other errors — stop cleanly
      userStoppedRef.current = true;
      setIsListening(false);
    };

    recognition.onend = () => {
      // If the user didn't manually stop, restart automatically
      // (browsers force-end the session every ~60s even with continuous=true)
      if (!userStoppedRef.current) {
        try {
          recognition.start();
          return;
        } catch (err) {
          console.warn('[VoiceInput] Auto-restart failed:', err);
        }
      }

      // User stopped OR restart failed — actually end the session
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
      // User wants to stop — set flag BEFORE calling stop()
      userStoppedRef.current = true;
      try {
        recognitionRef.current.stop();
      } catch (err) {
        console.warn('[VoiceInput] Stop error:', err);
      }
    } else {
      // Starting fresh — snapshot what's typed and reset flags
      const current = message.trim();
      prefixRef.current = current;
      finalBufferRef.current = '';
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

      {/* ─── VOICE INPUT BUTTON ─── */}
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
          {isListening ? (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="9" y="2" width="6" height="13" rx="3" fill="currentColor" />
              <path d="M19 10a7 7 0 01-14 0" />
              <line x1="12" y1="19" x2="12" y2="22" />
              <line x1="8" y1="22" x2="16" y2="22" />
            </svg>
          ) : (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="9" y="2" width="6" height="13" rx="3" />
              <path d="M19 10a7 7 0 01-14 0" />
              <line x1="12" y1="19" x2="12" y2="22" />
              <line x1="8" y1="22" x2="16" y2="22" />
            </svg>
          )}
        </button>
      )}

      {/* ─── SEND BUTTON ─── */}
      <button
        className="send-button"
        onClick={handleSubmit}
        disabled={!message.trim() || disabled}
      >
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  );
};
