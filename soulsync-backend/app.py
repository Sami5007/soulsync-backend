from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import torch
import shap
import re
import uuid
import random
import json
import threading
import os
from datetime import datetime
import logging
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# This tells Flask to let Vercel talk to it!
CORS(app, resources={r"/api/*": {"origins": "*"}})


print("Loading Lightweight Emotion model...")
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
emotion_model.eval()
print("Lightweight model loaded!")

emotion_labels = [emotion_model.config.id2label[i] for i in range(len(emotion_model.config.id2label))]
logger.info(f"Emotion labels from model config: {emotion_labels}")


# --- SHAP EXPLAINER ---
print("Initializing SHAP Explainer...")

def shap_predict(texts):
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
    elif isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(shap_predict, masker, output_names=emotion_labels)
print("SHAP Explainer ready!")


# Grok Model
GROK_MODEL = "grok-4-1-fast-reasoning"
logger.info(f"LLM Client ready — model: {GROK_MODEL}")


# --- CRISIS EMAIL SENDER ---
def send_crisis_email(user_message, severity, emotion, history):
    # 1. Grab the keys
    WEB3FORMS_KEY = os.getenv("WEB3FORMS_KEY")
    COUNSELOR_EMAIL = os.getenv("COUNSELOR_EMAIL")

    if not WEB3FORMS_KEY:
        logger.warning("Web3Forms key missing in secrets. Crisis email skipped.")
        return

    # 2. Extract recent history
    history_text = "\n".join(
        [f"{msg.get('sender', 'Unknown').capitalize()}: {msg.get('text', '')}" for msg in history[-5:]]
    ) if history else "No previous context."

    # 3. Build email body
    email_body = f"""
URGENT: Soul-Sync Crisis Alert

Message: "{user_message}"
Detected Emotion: {emotion}
Severity: {severity.upper()}

Recent Conversation Context:
{history_text}

Please review and intervene if necessary.
"""

    # 4. The Web3Forms Payload (Bypasses the Port 587 block)
    payload = {
        "access_key": WEB3FORMS_KEY,
        "subject": f"URGENT: Soul-Sync Critical Alert ({severity.upper()})",
        "from_name": "SoulSync AI Bot",
        "email": "noreply@soulsync.com",
        "message": email_body
    }

    try:
        # 5. Send the web request over standard port 443
        response = requests.post("https://api.web3forms.com/submit", json=payload, timeout=10)

        if response.status_code == 200:
            logger.info(f"Crisis email sent to {COUNSELOR_EMAIL or 'counselor via Web3Forms'}")
        else:
            logger.error(f"Failed to send email. API Response: {response.text}")
    except Exception as e:
        logger.error(f"Error triggering email API: {str(e)}")


def map_to_srs_emotions(detected_emotion):
    mapping = {
        "anger": "anger",
        "fear": "fear",
        "joy": "joy",
        "sadness": "sadness",
        "neutral": "neutral",
        "surprise": "confusion",
        "disgust": "anger"
    }
    return mapping.get(detected_emotion, "neutral")


RESPONSES = {}

def load_responses():
    global RESPONSES
    search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "responses.json"),
        os.path.join(os.getcwd(), "responses.json"),
        "responses.json"
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    RESPONSES = json.load(f)
                logger.info(f"responses.json loaded from: {path} — {len(RESPONSES)} emotions covered.")
                return
            except Exception as e:
                logger.error(f"Failed to parse responses.json: {e}")
    logger.warning("responses.json not found!")

load_responses()


# ====================== CRISIS DETECTION ======================
CRISIS_KEYWORDS = {
    "critical": [
        r"suicid", r"kill\s*(?:my\s*self|myself|myslef|myelf|me)",
        r"end\s*(?:it\s*all|my\s*life|everything)", r"take\s*my\s*(?:own\s*)?life",
        r"slit\s*(?:wrist|vein)", r"hang\s*(?:myself|me)",
        r"self\s*[\-\s]?harm", r"cut\s*myself", r"overdose",
    ],
    "high": [
        r"want\s*to\s*die", r"don'?t\s*want\s*to\s*(?:live|be\s*alive|exist)",
        r"no\s*(?:point|reason)\s*(?:in\s*)?(?:living|to\s*live|to\s*go\s*on)",
        r"better\s*off\s*(?:dead|without\s*me|if\s*i\s*(?:was|were)\s*gone)",
        r"wish\s*i\s*(?:was|were)\s*dead", r"can'?t\s*go\s*on",
        r"done\s*with\s*(?:life|everything|it\s*all)",
    ],
    "medium": [
        r"hopeless", r"worthless", r"useless",
        r"can'?t\s*(?:take|handle|cope|do\s*this)\s*(?:it|anymore|this)?",
        r"panic\s*attack", r"breaking\s*down", r"falling\s*apart",
        r"overwhelmed", r"stressed\s*out", r"can'?t\s*sleep",
    ]
}

def detect_crisis(message):
    message_lower = message.lower()
    found_severity = None
    keywords_found = []
    for level in ("critical", "high", "medium"):
        for pattern in CRISIS_KEYWORDS[level]:
            if re.search(pattern, message_lower, re.IGNORECASE):
                if found_severity is None:
                    found_severity = level
                keywords_found.append(pattern[:30])
        if found_severity == level:
            break
    return {
        "is_crisis": found_severity is not None,
        "severity": found_severity,
        "keywords_found": keywords_found[:3]
    }
# ============================================================


# ====================== HYBRID JSON + GROK ======================
def get_base_response(emotion, preference):
    """Returns ONE verified base response from responses.json"""
    if not RESPONSES or emotion not in RESPONSES:
        return None

    emotion_data = RESPONSES[emotion]

    if preference == "islamic":
        options = emotion_data.get("islamic", [])
    elif preference == "psychological":
        options = emotion_data.get("psychological", [])
    else:  # hybrid
        islamic = emotion_data.get("islamic", [])
        psychological = emotion_data.get("psychological", [])
        options = islamic + psychological

    if not options:
        return None
    return random.choice(options)


# ====================== SINGLE SOURCE OF TRUTH ======================
# Previously this function was defined 3 times — only the last one was active.
# Consolidated into ONE robust definition with all keywords merged.
def is_casual_message(message, emotion):
    """Smart detection: Is this normal/casual chat or emotional support needed?
    
    Returns True for:
    - Empty messages
    - Neutral emotion classifications
    - Messages containing greetings, simple questions, or casual chatter
    """
    if not message:
        return True

    msg_lower = message.lower().strip()

    casual_keywords = [
        # Greetings
        "hi", "hello", "hey", "assalamu alaikum", "walaikum", "salam",
        "good morning", "good night", "good evening",
        # Pleasantries
        "how are you", "how r u", "what's up", "sup",
        "thank you", "thanks", "shukriya", "bye", "goodbye",
        "ok", "okay", "cool",
        # Casual / off-topic queries
        "what should i", "what is", "how do i", "calculate", "2+2",
        "math", "random", "bored", "nothing", "just", "talking", "chat"
    ]

    # Casual if: neutral emotion OR contains casual keywords
    if emotion == "neutral" or any(kw in msg_lower for kw in casual_keywords):
        return True
    return False


def get_grok_response(message, emotion, preference, conversation_history):
    """CONTEXT-AWARE VERSION — remembers prior turns and references them."""

    is_casual = is_casual_message(message, emotion)

    if is_casual:
        system_prompt = """You are Soul-Sync, a warm, friendly Pakistani friend.

CONTEXT MEMORY RULES (CRITICAL):
- You MUST remember and reference what was discussed earlier in this conversation.
- If the user previously asked about CBT, Quranic verses, anxiety, breathing exercises, or any specific topic, recall it naturally.
- Use phrases like "as we discussed earlier", "going back to what you mentioned", "regarding that technique I shared".
- If the user says "yes", "more", "tell me more", "go on", "continue" — they are referring to your last message. Continue that exact topic.
- NEVER act like each message is a brand-new conversation.

STYLE:
Speak naturally and casually in clear English. Keep replies short (2-4 sentences).
Reference what the user just said. Never sound like a therapist."""
    else:
        base_response = get_base_response(emotion, preference)
        if not base_response:
            return get_fallback_response(emotion, preference)

        system_prompt = f"""You are Soul-Sync, a warm caring friend.

CONTEXT MEMORY RULES (CRITICAL):
- You MUST remember the entire conversation so far.
- If the user is following up on something you said before (e.g. "tell me more about that CBT technique", "yes please", "what was the second one") — continue the prior topic naturally.
- If the user already shared their problem in earlier turns, do NOT ask them to re-explain.
- Build on previous turns instead of treating each message as standalone.

User just said: "{message}"

Use this verified base response and rewrite it naturally,
weaving in the conversation context above:
{base_response}

Rules:
- First acknowledge what user said (and reference earlier turns if relevant).
- Keep exact meaning of base response.
- Sound like a real friend who has been listening throughout.
- Max 200 words."""

    # Build messages — full context window, last 10 turns (was 3 → memory issue fixed)
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for turn in conversation_history[-10:]:   # ↑ raised from 3 to 10
            role = "user" if turn.get("sender") == "user" else "assistant"
            text = turn.get('text', '').strip()
            if text:  # skip empty turns
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": message})

    try:
        api_key = os.getenv("GROK_KEY")
        if not api_key:
            return "Hey! I'm here. What's on your mind?" if is_casual else get_fallback_response(emotion, preference)

        response = requests.post(
            url="https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROK_MODEL,
                "messages": messages,
                "temperature": 0.65 if is_casual else 0.70,
                "max_tokens": 280,          # ↑ raised so context-aware replies aren't cut off
                "top_p": 0.9
            },
            timeout=10                       # ↑ 8 → 10s to accommodate longer context
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            logger.warning(f"Grok API returned {response.status_code}: {response.text}")
            return get_fallback_response(emotion, preference) if not is_casual else "Hey! How are you?"

    except Exception as e:
        logger.error(f"Grok error: {e}")
        return get_fallback_response(emotion, preference) if not is_casual else "I'm right here, friend."


def get_fallback_response(emotion, preference):
    if not RESPONSES or emotion not in RESPONSES:
        return "I'm here for you. Would you like to share more about how you're feeling?"
    emotion_data = RESPONSES[emotion]
    if preference == "islamic":
        options = emotion_data.get("islamic") or emotion_data.get("psychological")
    elif preference == "psychological":
        options = emotion_data.get("psychological") or emotion_data.get("islamic")
    else:
        mode = random.choice(["islamic", "psychological"])
        options = (emotion_data.get(mode) or emotion_data.get("islamic") or emotion_data.get("psychological"))
    if not options:
        return "I'm here for you. Would you like to share more about how you're feeling?"
    return random.choice(options)


def get_crisis_response(severity):
    if severity == "critical":
        return "I'm deeply concerned about your safety right now. Please reach out immediately:\nNational Mental Health Helpline: 1166\nEmergency Services: 1122\nYou are not alone."
    elif severity == "high":
        return "I hear you, and I'm genuinely concerned. Please consider speaking with someone:\nNational Mental Health Helpline: 1166 (24/7)\nYou don't have to carry this alone."
    else:
        return "It sounds like you're carrying something really heavy. I'm here with you. If things feel too overwhelming, please reach out — Helpline: 1166."
# ============================================================


# ====================== ROUTES ======================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "emotion_model": EMOTION_MODEL_NAME,
        "llm_model": GROK_MODEL,
        "emotions_supported": len(emotion_labels),
        "responses_loaded": len(RESPONSES) > 0,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        "name": "SOUL-SYNC",
        "version": "3.1",
        "description": "Muslim-Aware Mental Wellness Chatbot (Context-Aware)",
        "features": ["Emotion Detection", "Crisis Detection", "Hybrid JSON + Grok", "SHAP Explainability", "Conversation Memory"],
        "emotion_labels": emotion_labels
    }), 200


@app.route('/api/session/start', methods=['POST'])
def start_session():
    data = request.json or {}
    session_id = str(uuid.uuid4())
    logger.info(f"Stateless session initialized: {session_id}")
    return jsonify({
        "success": True,
        "session_id": session_id,
        "preference": data.get("preference", "hybrid")
    }), 201


@app.route('/api/preference', methods=['POST'])
def set_preference():
    try:
        data = request.json or {}
        preference = data.get("preference", "hybrid")
        if preference not in ("islamic", "psychological", "hybrid"):
            return jsonify({"error": "Invalid preference"}), 400
        return jsonify({"success": True, "preference": preference}), 200
    except Exception as e:
        logger.error(f"Error setting preference: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        message = data.get("message", "").strip()
        session_id = data.get("session_id")
        preference = data.get("preference", "hybrid")
        conversation_history = data.get("history", [])

        if not message:
            return jsonify({"error": "No message provided"}), 400
        if len(message) > 500:
            return jsonify({"error": "Message exceeds 500 character limit"}), 400

        crisis_result = detect_crisis(message)

        # Emotion Detection
        inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        logits = outputs.logits[0]
        scores = torch.nn.functional.softmax(logits, dim=0)
        top_emotion_idx = torch.argmax(scores).item()
        confidence = scores[top_emotion_idx].item()
        raw_emotion = emotion_labels[top_emotion_idx] if top_emotion_idx < len(emotion_labels) else "neutral"
        emotion = map_to_srs_emotions(raw_emotion)

        # SHAP
        try:
            shap_values = explainer([message])
            class_values = shap_values.values[0, :, top_emotion_idx]
            tokens = shap_values.data[0]
            word_impacts = []
            for token, impact in zip(tokens, class_values):
                clean_token = str(token).replace('Ġ', '').strip()
                if len(clean_token) > 0 and clean_token not in ['<s>', '</s>', '<pad>']:
                    word_impacts.append({"word": clean_token, "impact": round(float(impact), 4)})
            word_impacts = sorted(word_impacts, key=lambda x: x["impact"], reverse=True)
            top_shap_words = [w for w in word_impacts if w["impact"] > 0][:5]
        except Exception as e:
            logger.error(f"SHAP Error: {str(e)}")
            top_shap_words = []

        # ─── HYBRID PIPELINE: Crisis bypass FIRST, then Grok ───
        # (FIXED INDENTATION — was at column 0 causing IndentationError)
        if crisis_result["is_crisis"]:
            response_text = get_crisis_response(crisis_result["severity"])
            # Fire crisis email asynchronously (non-blocking) — backend redundancy
            threading.Thread(
                target=send_crisis_email,
                args=(message, crisis_result["severity"], emotion, conversation_history),
                daemon=True
            ).start()
        else:
            response_text = get_grok_response(
                message=message,
                emotion=emotion,
                preference=preference,
                conversation_history=conversation_history
            )

        logger.info(f"Emotion: {emotion} | Crisis: {crisis_result['is_crisis']} | Pref: {preference} | History turns: {len(conversation_history)}")

        return jsonify({
            "message": message,
            "emotion": emotion,
            "confidence": round(float(confidence), 4),
            "shap_values": top_shap_words,
            "response": response_text,
            "crisis": crisis_result,
            "session_id": session_id,
            "preference": preference,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        return jsonify({"error": "Something went wrong. Please try again.", "detail": str(e)}), 500


# Other routes (kept unchanged)
@app.route('/api/emotion/detect', methods=['POST'])
def emotion_detect():
    # Stub — keep your original logic if you have one
    pass


@app.route('/api/crisis/resources', methods=['GET'])
def crisis_resources():
    return jsonify({
        "pakistan": [
            {"name": "National Mental Health Helpline", "number": "1166", "available": "24/7"},
            {"name": "Rescue Emergency Services", "number": "1122", "available": "24/7"}
        ]
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == "__main__":
    logger.info("Starting SOUL-SYNC Backend v3.1 (Hybrid JSON + Grok + Context Memory)")
    app.run(debug=True, host='0.0.0.0', port=5000)
