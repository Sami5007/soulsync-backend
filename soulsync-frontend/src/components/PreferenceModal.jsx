import { useState } from 'react';
import './PreferenceModal.css';

export const PreferenceModal = ({ onConfirm, onCancel }) => {
  const [selected, setSelected] = useState('psychological');

  return (
    <div className="modal-overlay">
      <div className="preference-card">
        <button className="close-x" onClick={onCancel}>×</button>
        <h2 className="modal-title">Choose Your Guidance Mode</h2>
        <p className="modal-subtitle">Select the type of support that resonates with you. You can change this preference anytime.</p>

        <div className="option-group">
          <div 
            className={`pref-card-item ${selected === 'islamic' ? 'active' : ''}`}
            onClick={() => setSelected('islamic')}
          >
            <div className="pref-icon-box islamic">📖</div>
            <div className="pref-info">
              <strong>Islamic Guidance</strong>
              <span>Receive support through Quranic verses, Hadith references, and Islamic coping strategies aligned with your faith.</span>
            </div>
            <div className="radio-circle"></div>
          </div>

          <div 
            className={`pref-card-item ${selected === 'psychological' ? 'active' : ''}`}
            onClick={() => setSelected('psychological')}
          >
            <div className="pref-icon-box psychology">🧠</div>
            <div className="pref-info">
              <strong>Evidence-Based Psychology</strong>
              <span>Receive support through CBT techniques, mindfulness exercises, and scientifically validated therapeutic approaches.</span>
            </div>
            <div className="radio-circle"></div>
          </div>
        </div>

        <div className="modal-buttons">
          <button className="btn-cancel" onClick={onCancel}>Cancel</button>
          <button className="btn-continue" onClick={() => onConfirm(selected)}>Continue</button>
        </div>
      </div>
    </div>
  );
};