# main.py
"""
Nova Chat - Multi-LLM Intelligent Routing Chatbot
Main application entry point using Flask + SocketIO + Groq + Semantic Routing
"""

import os
import sys
from datetime import datetime

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from collections import defaultdict, deque

# ────────────────────────────────────────────────
# 1. Load environment variables FIRST
# ────────────────────────────────────────────────
load_dotenv()

# Basic validation
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY is required!")
    sys.exit(1)

# ────────────────────────────────────────────────
# 2. Create Flask application instance
# ────────────────────────────────────────────────
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'super-secret-key-change-me-please')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')  # ← Neon/PostgreSQL connection string
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
}

# ────────────────────────────────────────────────
# 3. Initialize extensions (order matters!)
# ────────────────────────────────────────────────
db = SQLAlchemy(app)
migrate = Migrate(app, db)          # This registers 'flask db' commands
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ────────────────────────────────────────────────
# 4. Import models AFTER db initialization
# ────────────────────────────────────────────────
# from models import ChatMessage, User  # ← import here when you create models.py

# ────────────────────────────────────────────────
# 5. LLM Models & Semantic Routing
# ────────────────────────────────────────────────
MODELS = {
    "model_xl": "openai/gpt-oss-safeguard-20b",  # ← placeholder, change when real
    "model_l": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "model_m": "meta-llama/llama-guard-4-12b",
    "model_s": "groq/compound-mini",             # fast & cheap fallback
}

ROUTES = {
    "model_xl": "Safety analysis, policy sensitive, high risk content",
    "model_l": "Complex reasoning, system design, deep explanations",
    "model_m": "Cybersecurity, vulnerabilities, exploits, moderation",
    "model_s": "Casual chat, greetings, short or simple questions",
}

router_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

ROUTE_EMBEDDINGS = {
    k: router_model.encode(v, normalize_embeddings=True)
    for k, v in ROUTES.items()
}

def route_prompt(prompt: str, threshold: float = 0.45) -> str:
    """Choose best model using semantic similarity"""
    prompt_emb = router_model.encode(prompt, normalize_embeddings=True)
    scores = {k: cosine_similarity([prompt_emb], [v])[0][0] for k, v in ROUTE_EMBEDDINGS.items()}
    best_model, best_score = max(scores.items(), key=lambda x: x[1])
    return best_model if best_score >= threshold else "model_s"

# ────────────────────────────────────────────────
# 6. LLM Call Helper
# ────────────────────────────────────────────────
def call_llm(model_key: str, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODELS[model_key],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ────────────────────────────────────────────────
# 7. Routes
# ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ────────────────────────────────────────────────
# 8. SocketIO Event Handlers
# ────────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print(f"Client connected → SID: {request.sid}")
    emit('chat_message', {
        'role': 'system',
        'content': 'Welcome to Nova Chat • Ready when you are!'
    })

@socketio.on('message')
def handle_message(data):
    user_message = data.get('message', '').strip()
    if not user_message:
        return

    # Broadcast user message
    emit('chat_message', {
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.utcnow().isoformat()
    }, broadcast=True)

    # Route to best model
    model_key = route_prompt(user_message)

    emit('typing', {'status': True}, broadcast=True)

    try:
        answer = call_llm(model_key, user_message)
        emit('chat_message', {
            'role': 'assistant',
            'content': answer,
            'timestamp': datetime.utcnow().isoformat(),
            'model': model_key  # optional
        }, broadcast=True)
    except Exception as e:
        emit('chat_message', {
            'role': 'assistant',
            'content': f"Error: {str(e)}"
        }, broadcast=True)
    finally:
        emit('typing', {'status': False}, broadcast=True)

# ────────────────────────────────────────────────
# 9. Run the application
# ────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("   Nova Chat - Multi-LLM Routing Chatbot")
    print("   Open → http://127.0.0.1:5000")
    print("=" * 70)
    
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        allow_unsafe_werkzeug=True
    )


# ────────────────────────────────────────────────
# Global in-memory storage: sid → deque of messages (limited size)
# Each message: {"role": "user"|"assistant", "content": str, "timestamp": str}
# ────────────────────────────────────────────────
chat_histories = defaultdict(lambda: deque(maxlen=10))  # max 10 messages → ~5 turns


def get_history_for_llm(sid: str) -> list[dict]:
    """Returns last N messages in OpenAI-compatible format"""
    return [{"role": msg["role"], "content": msg["content"]} for msg in chat_histories[sid]]


def add_to_history(sid: str, role: str, content: str):
    """Add message and keep only the newest ones"""
    chat_histories[sid].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })


# ────────────────────────────────────────────────
# Optional: cleanup very old sessions (every 100 connections, for example)
# You can also run this in a background thread if desired
# ────────────────────────────────────────────────
def cleanup_old_histories():
    to_remove = []
    for sid, hist in list(chat_histories.items()):
        if len(hist) == 0 or (datetime.utcnow() - datetime.fromisoformat(hist[-1]["timestamp"])).total_seconds() > 3600*24:  # > 24h
            to_remove.append(sid)
    for sid in to_remove:
        del chat_histories[sid]


# ────────────────────────────────────────────────
# Modified SocketIO handlers
# ────────────────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print(f"Client connected → SID: {request.sid}")
    # Optional: send welcome + previous messages if any (usually empty on first connect)
    history = get_history_for_llm(request.sid)
    if history:
        for msg in history[-4:]:  # last few only, avoid flooding
            emit('chat_message', {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': "past"
            })
    emit('chat_message', {
        'role': 'system',
        'content': 'Welcome to Nova Chat • Ready when you are!'
    })


@socketio.on('message')
def handle_message(data):
    user_message = data.get('message', '').strip()
    if not user_message:
        return

    sid = request.sid

    # 1. Add user message to history & broadcast
    add_to_history(sid, "user", user_message)
    emit('chat_message', {
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.utcnow().isoformat()
    }, broadcast=True)   # ← still broadcast (public feel), or remove if private chat

    # 2. Route → select model
    model_key = route_prompt(user_message)

    emit('typing', {'status': True}, broadcast=True)

    try:
        # 3. Prepare full context: system + history + current user message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        messages.extend(get_history_for_llm(sid))           # ← this is the key change!
        messages.append({"role": "user", "content": user_message})

        # 4. Call LLM with full context
        response = client.chat.completions.create(
            model=MODELS[model_key],
            messages=messages,   # ← now includes history
            temperature=0.7,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content.strip()

        # 5. Save assistant reply
        add_to_history(sid, "assistant", answer)

        # 6. Send to client
        emit('chat_message', {
            'role': 'assistant',
            'content': answer,
            'timestamp': datetime.utcnow().isoformat(),
            'model': model_key
        }, broadcast=True)

    except Exception as e:
        emit('chat_message', {
            'role': 'assistant',
            'content': f"Error: {str(e)}"
        }, broadcast=True)

    finally:
        emit('typing', {'status': False}, broadcast=True)

    # Optional: clean up very old sessions every now and then
    if len(chat_histories) % 100 == 0:
        cleanup_old_histories()


# Optional: disconnect handler (clean up if you want strict cleanup)
@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected → SID: {request.sid}")
    # You can remove history here if you want sessions to be forgotten on disconnect
    # del chat_histories[request.sid]   # ← uncomment if desired