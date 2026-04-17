export async function sendCrisisEmail(message, severity, emotion, history) {
  const historyText = history
    .slice(-5)
    .map(m => `${m.type === 'user' ? 'User' : 'Bot'}: ${m.text}`)
    .join('\n');

  const payload = {
    access_key: "ea2198c1-442c-41e0-9fd2-68d6966b76f3",
    subject: `🚨 URGENT: Soul-Sync Crisis Alert (${severity.toUpperCase()})`,
    from_name: "SoulSync Crisis System",
    email: "soulsync.alerts@gmail.com",
    message: `
URGENT: Soul-Sync Crisis Alert

Timestamp: ${new Date().toLocaleString()}
Message: "${message}"
Detected Emotion: ${emotion}
Severity: ${severity.toUpperCase()}

Recent Conversation:
${historyText}

Please review and intervene if necessary.
Pakistan Mental Health Helpline: 1166
Emergency Services: 1122
    `
  };

  try {
    const res = await fetch("https://api.web3forms.com/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      console.log("Crisis email sent successfully");
    } else {
      console.error("Crisis email failed:", data);
    }
  } catch (err) {
    console.error("Crisis email error:", err);
  }
}
