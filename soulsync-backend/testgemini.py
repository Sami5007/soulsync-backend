
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("SOUL-SYNC DIAGNOSTIC TEST")
print("=" * 50)

# ── TEST 1: API KEY ──────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"[PASS] GEMINI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
else:
    print("[FAIL] GEMINI_API_KEY not found — check your .env file")
    print("       Make sure .env contains:  GEMINI_API_KEY=your_key_here")
    exit(1)

# ── TEST 2: GEMINI IMPORT ────────────────────────
try:
    from google import genai
    from google.genai import types
    print("[PASS] google-genai package imported successfully")
except ImportError as e:
    print(f"[FAIL] google-genai import failed: {e}")
    print("       Run: pip install google-genai")
    exit(1)

# ── TEST 3: GEMINI API CALL ──────────────────────
try:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Say hello in one sentence as a mental wellness chatbot.",
        config=types.GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.7
        )
    )
    print(f"[PASS] Gemini API call successful!")
    print(f"       Response: {response.text.strip()}")
except Exception as e:
    print(f"[FAIL] Gemini API call failed: {e}")
    exit(1)

# ── TEST 4: CRISIS DETECTION LOGIC ──────────────
import re

CRISIS_KEYWORDS = {
    "critical": [
        r"suicid",
        r"kill\s*(?:myself|me)",
        r"end\s*(?:it\s*all|my\s*life|everything)",
        r"take\s*my\s*(?:own\s*)?life",
        r"slit\s*(?:wrist|vein)",
        r"hang\s*(?:myself|me)",
        r"self\s*[\-\s]?harm",
        r"cut\s*myself",
        r"overdose",
    ],
    "high": [
        r"want\s*to\s*die",
        r"don'?t\s*want\s*to\s*(?:live|be\s*alive|exist)",
        r"no\s*(?:point|reason)\s*(?:in\s*)?(?:living|to\s*live|to\s*go\s*on)",
        r"better\s*off\s*(?:dead|without\s*me|if\s*i\s*(?:was|were)\s*gone)",
        r"wish\s*i\s*(?:was|were)\s*dead",
        r"can'?t\s*go\s*on",
        r"done\s*with\s*(?:life|everything|it\s*all)",
    ],
    "medium": [
        r"hopeless",
        r"worthless",
        r"useless",
        r"can'?t\s*(?:take|handle|cope|do\s*this)\s*(?:it|anymore|this)?",
        r"panic\s*attack",
        r"breaking\s*down",
        r"falling\s*apart",
        r"overwhelmed",
        r"stressed\s*out",
        r"can'?t\s*sleep",
    ]
}

test_messages = [
    ("i want to kill myself",   "critical"),
    ("i want to die",           "high"),
    ("dont want to live",       "high"),
    ("i feel hopeless",         "medium"),
    ("i am overwhelmed",        "medium"),
    ("i am happy today",        None),
    ("hey",                     None),
]

print("\n── Crisis Detection Tests ──────────────────")
all_passed = True
for message, expected_severity in test_messages:
    msg_lower = message.lower()
    found = None
    for level in ("critical", "high", "medium"):
        for pattern in CRISIS_KEYWORDS[level]:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                found = level
                break
        if found:
            break

    status = "PASS" if found == expected_severity else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  [{status}] '{message}' → expected: {expected_severity}, got: {found}")

if all_passed:
    print("\n[PASS] All crisis detection tests passed!")
else:
    print("\n[WARN] Some crisis tests failed — check patterns above")

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)