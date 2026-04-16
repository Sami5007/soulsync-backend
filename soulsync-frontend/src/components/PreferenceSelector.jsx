import { useState, useRef, useEffect } from 'react';
import './PreferenceSelector.css';

export const PreferenceSelector = ({ preference, onPreferenceChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const options = [
    { id: 'hybrid', icon: '⚖️', label: 'Hybrid Mode', desc: 'Balanced support' },
    { id: 'islamic', icon: '🕌', label: 'Islamic', desc: 'Quran & Sunnah' },
    { id: 'psychological', icon: '🧠', label: 'Psychological', desc: 'Therapy techniques' }
  ];

  const currentOption = options.find(o => o.id === preference) || options[0];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="pref-dropdown-container" ref={dropdownRef}>
      <button className="pref-active-btn" onClick={() => setIsOpen(!isOpen)}>
        <span className="pref-icon">{currentOption.icon}</span>
        <span className="pref-label">{currentOption.label}</span>
        <svg className={`pref-arrow ${isOpen ? 'open' : ''}`} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </button>

      {isOpen && (
        <div className="pref-menu">
          {options.map((opt) => (
            <button
              key={opt.id}
              className={`pref-menu-item ${preference === opt.id ? 'selected' : ''}`}
              onClick={() => {
                onPreferenceChange(opt.id);
                setIsOpen(false);
              }}
            >
              <span className="pref-icon">{opt.icon}</span>
              <div className="pref-text">
                <div className="pref-title">{opt.label}</div>
                <div className="pref-desc">{opt.desc}</div>
              </div>
              {preference === opt.id && <span className="pref-check">✓</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};