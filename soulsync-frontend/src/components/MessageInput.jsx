import { useState, useRef } from 'react';
import './MessageInput.css';

export const MessageInput = ({ onSendMessage, disabled }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);
  const MAX_CHARS = 500;

  const handleSubmit = async () => {
    if (!message.trim() || disabled || message.length > MAX_CHARS) return;
    const text = message.trim();
    setMessage('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
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
          placeholder={disabled ? "Waiting for connection..." : "Type your message here..."}
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
      <button 
        className="send-button" 
        onClick={handleSubmit} 
        disabled={!message.trim() || disabled}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  );
};