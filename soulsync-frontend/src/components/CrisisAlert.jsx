import { useState } from 'react';
import './CrisisAlert.css';

export const CrisisAlert = ({ isCrisis, crisisData }) => {
  const [dismissed, setDismissed] = useState(false);
  if (!isCrisis || dismissed) return null;

  return (
    <div className={`crisis-alert severity-${crisisData?.severity}`}>
      <div className="crisis-content">
        <h4>Support is available</h4>
        <p>I'm concerned about what you're sharing. Please reach out to professional help:</p>
        <div className="helpline-links">
          <a href="tel:1166" className="helpline-btn">Call 1166 (Mental Health)</a>
          <a href="tel:1122" className="helpline-btn emergency">Call 1122 (Emergency)</a>
        </div>
        <button onClick={() => setDismissed(true)} className="close-alert">Dismiss</button>
      </div>
    </div>
  );
};