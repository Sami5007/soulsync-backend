import { useState } from 'react';
import './CrisisConsentmodal.css';

export const CrisisConsentmodal = ({ crisisData, onConsent, onDecline }) => {
  const [sending, setSending] = useState(false);

  const severityConfig = {
    critical: {
      color: '#dc2626',
      bg: 'rgba(220, 38, 38, 0.15)',
      border: 'rgba(220, 38, 38, 0.4)',
      label: 'CRITICAL',
      message: 'We are very concerned about your safety right now.',
    },
    high: {
      color: '#f59e0b',
      bg: 'rgba(245, 158, 11, 0.15)',
      border: 'rgba(245, 158, 11, 0.4)',
      label: 'HIGH',
      message: 'It sounds like you may be going through something very difficult.',
    },
    medium: {
      color: '#facc15',
      bg: 'rgba(250, 204, 21, 0.15)',
      border: 'rgba(250, 204, 21, 0.4)',
      label: 'MEDIUM',
      message: 'It seems like you might be experiencing some distress.',
    },
  };

  const severity = crisisData?.severity || 'medium';
  const config = severityConfig[severity] || severityConfig.medium;

  const handleConsent = async () => {
    setSending(true);
    await onConsent();
    setSending(false);
  };

  return (
    <div className="crisis-consent-overlay">
      <div className="crisis-consent-card">
        {/* Severity badge */}
        <div
          className="crisis-consent-badge"
          style={{ background: config.bg, border: `1px solid ${config.border}`, color: config.color }}
        >
          ⚠️ {config.label} CONCERN DETECTED
        </div>

        {/* Main message */}
        <h2 className="crisis-consent-title">{config.message}</h2>

        <p className="crisis-consent-desc">
          Would you like us to <strong>notify a professional counselor</strong> about 
          this conversation? They can reach out to offer support.
        </p>

        {/* What gets sent */}
        <div className="crisis-consent-info">
          <div className="crisis-consent-info-title">What will be shared:</div>
          <ul>
            <li>Your recent conversation context (last 5 messages)</li>
            <li>The detected emotion and severity level</li>
            <li>Timestamp of the alert</li>
          </ul>
          <div className="crisis-consent-info-note">
            🔒 No personal identifying information is included.
          </div>
        </div>

        {/* Action buttons */}
        <div className="crisis-consent-actions">
          <button
            className="crisis-consent-yes"
            onClick={handleConsent}
            disabled={sending}
          >
            {sending ? (
              <span className="crisis-consent-spinner" />
            ) : (
              <>✅ Yes, notify a counselor</>
            )}
          </button>

          <button
            className="crisis-consent-no"
            onClick={onDecline}
            disabled={sending}
          >
            No, I'm okay for now
          </button>
        </div>

        {/* Always-visible helplines */}
        <div className="crisis-consent-helplines">
          <p>You can always reach out directly:</p>
          <div className="crisis-consent-numbers">
            <a href="tel:1166" className="crisis-consent-hotline">
              <span>📞</span> Mental Health Helpline: <strong>1166</strong>
            </a>
            <a href="tel:1122" className="crisis-consent-hotline">
              <span>🚑</span> Emergency Services: <strong>1122</strong>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};
