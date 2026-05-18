import sqlite3
from email.mime import message
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
import time
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://soulsync-frontend-blue.vercel.app",
    "*"
]}})


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


CLAUDE_MODEL = "claude-haiku-4-5"
logger.info(f"LLM Client ready — model: {CLAUDE_MODEL}")

# ═══════════════════════════════════════════════════════════════
# ADMIN ANALYTICS — SQLite Setup
# ═══════════════════════════════════════════════════════════════
ANALYTICS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.db")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminsoulsync099117")
ACTIVE_ADMIN_TOKENS = {}
ADMIN_TOKEN_TTL_SECONDS = 60 * 60 * 4


@contextmanager
def get_db():
    conn = sqlite3.connect(ANALYTICS_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_analytics_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id_short TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                emotion TEXT,
                confidence REAL,
                preference TEXT,
                is_crisis INTEGER DEFAULT 0,
                crisis_severity TEXT,
                response_time_ms INTEGER,
                hour INTEGER,
                day_of_week INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_analytics(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emotion ON chat_analytics(emotion)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_crisis ON chat_analytics(is_crisis)")
        logger.info(f"Analytics DB initialized at {ANALYTICS_DB_PATH}")


def log_analytics(session_id, emotion, confidence, preference, crisis_result, response_time_ms):
    try:
        now = datetime.now()
        session_short = (session_id or "anon")[-4:] if session_id else "anon"
        with get_db() as conn:
            conn.execute("""
                INSERT INTO chat_analytics
                (session_id_short, emotion, confidence, preference,
                 is_crisis, crisis_severity, response_time_ms, hour, day_of_week)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_short,
                emotion,
                round(float(confidence), 4),
                preference,
                1 if crisis_result.get("is_crisis") else 0,
                crisis_result.get("severity"),
                response_time_ms,
                now.hour,
                now.weekday()
            ))
    except Exception as e:
        logger.error(f"Analytics log failed: {e}")


def verify_admin_token(token):
    if not token or token not in ACTIVE_ADMIN_TOKENS:
        return False
    expiry = ACTIVE_ADMIN_TOKENS[token]
    if time.time() > expiry:
        del ACTIVE_ADMIN_TOKENS[token]
        return False
    return True


def require_admin(req):
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return verify_admin_token(token)
    return False


init_analytics_db()
# ═══════════════════════════════════════════════════════════════


# --- CRISIS EMAIL SENDER ---
def send_crisis_email(user_message, severity, emotion, history):
    WEB3FORMS_KEY = os.getenv("WEB3FORMS_KEY")
    COUNSELOR_EMAIL = os.getenv("COUNSELOR_EMAIL")

    if not WEB3FORMS_KEY:
        logger.warning("Web3Forms key missing in secrets. Crisis email skipped.")
        return

    history_text = "\n".join(
        [f"{msg.get('sender', 'Unknown').capitalize()}: {msg.get('text', '')}" for msg in history[-5:]]
    ) if history else "No previous context."

    email_body = f"""
URGENT: Soul-Sync Crisis Alert

Message: "{user_message}"
Detected Emotion: {emotion}
Severity: {severity.upper()}

Recent Conversation Context:
{history_text}

Please review and intervene if necessary.
"""

    payload = {
        "access_key": WEB3FORMS_KEY,
        "subject": f"URGENT: Soul-Sync Critical Alert ({severity.upper()})",
        "from_name": "SoulSync AI Bot",
        "email": "noreply@soulsync.com",
        "message": email_body
    }

    try:
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


# ====================== CRISIS KEYWORDS ======================
CRISIS_KEYWORDS = {
    "critical": [
        r"suicid",
        r"commit\s*suicide", r"committing\s*suicide",
        r"kill\s*(?:my\s*self|myself|myslef|myelf|mself|me)",
        r"killing\s*(?:myself|me)",
        r"end\s*(?:it\s*all|my\s*life|everything|myself|me)",
        r"ending\s*(?:my\s*life|it\s*all|myself)",
        r"take\s*my\s*(?:own\s*)?life",
        r"taking\s*my\s*(?:own\s*)?life",
        r"end\s*it\s*tonight",
        r"slit\s*(?:my\s*)?(?:wrist|wrists|vein|veins|throat)",
        r"cut\s*(?:my\s*)?(?:wrist|wrists|vein|veins|throat)",
        r"hang\s*(?:myself|me|my\s*self)",
        r"hanging\s*(?:myself|me)",
        r"shoot\s*(?:myself|my\s*self|me)",
        r"jump\s*(?:off|from)\s*(?:a\s*|the\s*)?(?:bridge|building|roof|terrace)",
        r"jumping\s*off",
        r"step\s*in\s*front\s*of\s*(?:a\s*)?(?:train|car|truck)",
        r"drink\s*(?:bleach|poison|acid)",
        r"swallow\s*(?:pills|poison)",
        r"self\s*[\-\s]?harm", r"self\s*[\-\s]?harming",
        r"hurt\s*(?:myself|my\s*self)",
        r"harm\s*(?:myself|my\s*self)",
        r"cut\s*(?:myself|my\s*self)",
        r"cutting\s*(?:myself|my\s*self)",
        r"burn\s*(?:myself|my\s*self)",
        r"overdose", r"od\s*on\s*(?:pills|drugs)",
        r"take\s*(?:all\s*)?(?:the\s*)?pills",
        r"khud[\s\-]?kushi", r"khudkushi", r"khud\s*kashi",
        r"khudkhushi", r"khud[\s\-]?khushi",
        r"khud\s*ko\s*mar(?:na|do|donga|loon|loungi)",
        r"apne\s*aap\s*ko\s*mar",
        r"main\s*marn?a\s*chahta",
        r"mein\s*marn?a\s*chahti",
        r"mujhe\s*marn?a\s*hai",
        r"mujhay\s*marn?a\s*hai",
        r"mar\s*jaun(?:ga|gi)?",
        r"mar\s*jana\s*chahta",
        r"mar\s*jana\s*chahti",
        r"jaan\s*de\s*(?:du(?:nga|ngi)|donga|dungi|du)",
        r"jaan\s*lena",
        r"apni\s*jaan\s*(?:lena|le\s*lunga|le\s*lungi)",
        r"zindagi\s*khatam\s*kar",
        r"zindagi\s*khatm\s*kar",
        r"life\s*khatam\s*kar",
        r"phaansi\s*(?:laga|lagana|le\s*lunga)",
        r"phansi\s*(?:laga|lagana)",
        r"nas\s*kaat",
        r"nas\s*katna", r"nasein\s*kaat",
        r"haath\s*ki\s*nas",
        r"zehar\s*(?:kha|khana|pee|peena)",
        r"zaher\s*kha",
        r"chhat\s*se\s*kood",
        r"chat\s*se\s*kood",
        r"goli\s*mar",
        r"train\s*ke\s*aage",
    ],

    "high": [
        r"want\s*to\s*die", r"wanna\s*die",
        r"don'?t\s*want\s*to\s*(?:live|be\s*alive|exist|be\s*here)",
        r"do\s*not\s*want\s*to\s*(?:live|exist)",
        r"no\s*(?:point|reason)\s*(?:in\s*)?(?:living|to\s*live|to\s*go\s*on|to\s*be\s*alive)",
        r"better\s*off\s*(?:dead|without\s*me|if\s*i\s*(?:was|were)\s*gone|if\s*i\s*didn'?t\s*exist)",
        r"world\s*(?:would\s*be\s*)?better\s*without\s*me",
        r"everyone\s*(?:would\s*be\s*)?better\s*off\s*without\s*me",
        r"wish\s*i\s*(?:was|were)\s*dead",
        r"wish\s*i\s*(?:was|were)\s*never\s*born",
        r"wish\s*i\s*didn'?t\s*exist",
        r"wish\s*i\s*could\s*(?:disappear|vanish|just\s*sleep\s*forever)",
        r"can'?t\s*go\s*on",
        r"cannot\s*go\s*on",
        r"done\s*with\s*(?:life|everything|it\s*all|this\s*world)",
        r"tired\s*of\s*(?:living|being\s*alive|life|everything)",
        r"ready\s*to\s*(?:die|end\s*it)",
        r"goodbye\s*(?:forever|cruel\s*world|everyone)",
        r"saying\s*goodbye",
        r"won'?t\s*be\s*here\s*tomorrow",
        r"this\s*is\s*(?:my\s*)?(?:last|final)\s*(?:message|goodbye|night)",
        r"marn?a\s*chahta\s*hoon", r"marn?a\s*chahti\s*hoon",
        r"marn?a\s*chahta\s*hu", r"marn?a\s*chahti\s*hu",
        r"marn?a\s*chahti?\s*ho",
        r"jeena\s*nahi\s*chahta", r"jeena\s*nahi\s*chahti",
        r"jeena\s*nahin\s*chahta", r"jeena\s*nahin\s*chahti",
        r"jeena\s*mushkil",
        r"zinda\s*nahi\s*rehna", r"zinda\s*nahin\s*rehna",
        r"zindagi\s*se\s*tang",
        r"zindagi\s*se\s*thak",
        r"zindagi\s*bekaar",
        r"meri\s*zindagi\s*ka\s*koi\s*matlab",
        r"jeene\s*ka\s*koi\s*(?:matlab|maqsad|faida)",
        r"jeene\s*ki\s*koi\s*wajah",
        r"agar\s*main\s*na\s*hota", r"agar\s*mein\s*na\s*hoti",
        r"sab\s*meri\s*wajah\s*se",
        r"meray\s*bina\s*sab\s*better",
        r"thak\s*gaya\s*hoon", r"thak\s*gayi\s*hoon",
        r"bardasht\s*nahi\s*hota", r"bardasht\s*nahin\s*hoti",
        r"alvida\s*hamesha", r"akhri\s*(?:message|paigham|baat)",
    ],

    "medium": [
        r"hopeless", r"helplessness?",
        r"worthless", r"useless",
        r"i\s*(?:am|'m)\s*(?:a\s*)?(?:burden|failure|loser)",
        r"nobody\s*(?:cares|loves\s*me|would\s*miss\s*me)",
        r"no\s*one\s*(?:cares|understands|loves\s*me)",
        r"i\s*hate\s*(?:myself|my\s*life)",
        r"i\s*can'?t\s*do\s*this\s*anymore",
        r"can'?t\s*(?:take|handle|cope|deal\s*with|do\s*this)\s*(?:it|anymore|this|life|things)?",
        r"cannot\s*(?:take|handle|cope)",
        r"empty\s*inside", r"feel\s*empty", r"feel\s*nothing", r"feel\s*numb",
        r"panic\s*attack", r"panic\s*attacks",
        r"anxiety\s*attack",
        r"breaking\s*down", r"breakdown",
        r"falling\s*apart", r"falling\s*to\s*pieces",
        r"can'?t\s*breathe",
        r"overwhelmed", r"overwhelming",
        r"stressed\s*out", r"super\s*stressed",
        r"can'?t\s*sleep", r"haven'?t\s*slept", r"insomnia",
        r"crying\s*all\s*the\s*time", r"can'?t\s*stop\s*crying",
        r"in\s*so\s*much\s*pain", r"unbearable\s*pain",
        r"trapped", r"stuck",
        r"depressed", r"depression",
        r"udaas", r"udaasi", r"bohat\s*udaas", r"bahut\s*udaas",
        r"depress(?:ed|ion)?\s*(?:hoon|hu|hai)",
        r"pareshan(?:i)?", r"bohat\s*pareshan", r"bahut\s*pareshan",
        r"tanha(?:i)?",
        r"akela", r"akeli",
        r"koi\s*nahi\s*samajhta", r"koi\s*nahin\s*samajhta",
        r"koi\s*nahi\s*samjhta",
        r"koi\s*meri\s*parwa\s*nahi", r"koi\s*meri\s*parwa\s*nahin",
        r"sab\s*khilaaf",
        r"ghutan",
        r"dam\s*ghut",
        r"ro\s*raha\s*hoon", r"ro\s*rahi\s*hoon",
        r"rona\s*nahi\s*ruk",
        r"neend\s*nahi\s*aati", r"neend\s*nahin\s*aati",
        r"neend\s*nahi\s*aa\s*rahi",
        r"so\s*nahi\s*pa\s*raha", r"so\s*nahi\s*pa\s*rahi",
        r"thak\s*chuka", r"thak\s*chuki",
        r"himmat\s*nahi", r"himmat\s*nahin",
        r"himmat\s*tut",
        r"khaali\s*khaali",
        r"andar\s*se\s*tut",
        r"tut\s*chuka", r"tut\s*chuki",
        r"bekaar\s*hoon", r"nikamma\s*hoon", r"nikammi\s*hoon",
        r"main\s*kuch\s*nahi", r"mein\s*kuch\s*nahi",
        r"sab\s*meri\s*ghalti",
        r"bohat\s*tension", r"bahut\s*tension",
        r"dimagh\s*kharab",
    ]
}


# ====================== DETECT CRISIS FUNCTION ======================
def detect_crisis(message):
    """Scan message against crisis keywords and return severity."""
    if not message:
        return {"is_crisis": False, "severity": None}

    msg_lower = message.lower()

    for severity in ["critical", "high", "medium"]:
        for pattern in CRISIS_KEYWORDS.get(severity, []):
            if re.search(pattern, msg_lower):
                return {
                    "is_crisis": True,
                    "severity": severity
                }

    return {"is_crisis": False, "severity": None}
# ============================================================


# ====================== HYBRID JSON + CLAUDE ======================
def get_base_response(emotion, preference):
    if not RESPONSES or emotion not in RESPONSES:
        return None

    emotion_data = RESPONSES[emotion]

    if preference == "islamic":
        options = emotion_data.get("islamic", [])
    elif preference == "psychological":
        options = emotion_data.get("psychological", [])
    else:
        islamic = emotion_data.get("islamic", [])
        psychological = emotion_data.get("psychological", [])
        options = islamic + psychological

    if not options:
        return None
    return random.choice(options)


def is_casual_message(message, emotion):
    if not message:
        return True

    msg_lower = message.lower().strip()

    casual_keywords = [
        "hi", "hello", "hey", "assalamu alaikum", "walaikum", "salam",
        "good morning", "good night", "good evening",
        "how are you", "how r u", "what's up", "sup",
        "thank you", "thanks", "shukriya", "bye", "goodbye",
        "ok", "okay", "cool",
        "what should i", "what is", "how do i", "calculate", "2+2",
        "math", "random", "bored", "nothing", "just", "talking", "chat"
    ]

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
Speak naturally and casually in clear English. 
STRICT LENGTH: Maximum 2 or 5 short sentences. Be direct and concise. 
Reference what the user just said. Never sound like a therapist. Never sound like a therapist."""
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
Rules:
- First acknowledge what user said (and reference earlier turns if relevant).
- Keep exact meaning of base response.
- Sound like a real friend who has been listening throughout.
- STRICT LENGTH LIMIT: Maximum 2 to 3 short sentences.
- Be incredibly concise and straight to the point. Do not write essays or long paragraphs."""

    # ✅ FIXED: Initialize an empty array. No system roles here!
    messages = []

    if conversation_history:
        for turn in conversation_history[-10:]:
            role = "user" if turn.get("sender") == "user" else "assistant"
            text = turn.get('text', '').strip()
            if text:
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": message})

    try:
        api_key = os.getenv("CLAUDE_KEY")
        if not api_key:
            return "Hey! I'm here. What's on your mind?" if is_casual else get_fallback_response(emotion, preference)

        response = requests.post(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": CLAUDE_MODEL,
                "system": system_prompt, # ✅ System prompt passed correctly here
                "messages": messages,    # ✅ Clean array with only user/assistant
                "temperature": 0.65 if is_casual else 0.70,
                "max_tokens": 280,
            
            },
            timeout=10
        )

        if response.status_code == 200:
            response_data = response.json()
            content_blocks = response_data.get('content', [])
            final_text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    final_text += block.get("text", "")
            return final_text.strip()
        else:
            logger.warning(f"Claude API returned {response.status_code}: {response.text}")
            return get_fallback_response(emotion, preference) if not is_casual else "Hey! How are you?"

    except Exception as e:
        logger.error(f"Claude error: {e}")
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
        "llm_model": CLAUDE_MODEL,
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
        "features": ["Emotion Detection", "Crisis Detection", "Hybrid JSON + Claude", "SHAP Explainability", "Conversation Memory"],
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
    start_ts = time.time()
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        message = data.get("message", "").strip()
        session_id = data.get("session_id")
        preference = data.get("preference", "hybrid")
        conversation_history = data.get("history", [])

        if conversation_history and conversation_history[-1].get("text") == message:
            conversation_history = conversation_history[:-1]

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

        # ─── HYBRID PIPELINE ───
        if crisis_result["is_crisis"]:
            response_text = get_crisis_response(crisis_result["severity"])
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

        response_time_ms = int((time.time() - start_ts) * 1000)
        log_analytics(session_id, emotion, confidence, preference, crisis_result, response_time_ms)

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


@app.route('/api/emotion/detect', methods=['POST'])
def emotion_detect():
    try:
        data = request.json or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "No message provided"}), 400
        inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        logits = outputs.logits[0]
        scores = torch.nn.functional.softmax(logits, dim=0)
        top_idx = torch.argmax(scores).item()
        emotion = map_to_srs_emotions(emotion_labels[top_idx])
        confidence = scores[top_idx].item()
        return jsonify({"emotion": emotion, "confidence": round(float(confidence), 4)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/crisis/resources', methods=['GET'])
def crisis_resources():
    return jsonify({
        "pakistan": [
            {"name": "National Mental Health Helpline", "number": "1166", "available": "24/7"},
            {"name": "Rescue Emergency Services", "number": "1122", "available": "24/7"}
        ]
    }), 200


@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    try:
        data = request.json or {}
        password = data.get("password", "")
        if password == ADMIN_PASSWORD:
            token = str(uuid.uuid4())
            ACTIVE_ADMIN_TOKENS[token] = time.time() + ADMIN_TOKEN_TTL_SECONDS
            return jsonify({"success": True, "token": token}), 200
        return jsonify({"success": False, "error": "Invalid password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/verify', methods=['GET'])
def admin_verify():
    return jsonify({"valid": require_admin(request)}), 200


@app.route('/api/admin/analytics', methods=['GET'])
def admin_analytics():
    if not require_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM chat_analytics ORDER BY timestamp DESC LIMIT 200").fetchall()
            data = [dict(row) for row in rows]
        return jsonify({"analytics": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/crisis/send-email', methods=['POST'])
def crisis_send_email():
    try:
        data = request.json or {}
        threading.Thread(
            target=send_crisis_email,
            args=(
                data.get("message", ""),
                data.get("severity", "medium"),
                data.get("emotion", "unknown"),
                data.get("history", [])
            ),
            daemon=True
        ).start()
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == "__main__":
    logger.info("Starting SOUL-SYNC Backend v3.1 (Hybrid JSON + Claude + Context Memory)")
    app.run(debug=True, host='0.0.0.0', port=5000)
