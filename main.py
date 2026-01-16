# app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-change-me'
socketio = SocketIO(app, cors_allowed_origins="*")

# Check API key
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in environment!")
    exit(1)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ────────────────────────────────────────────────
# Models & Routing Configuration
# ────────────────────────────────────────────────
MODELS = {
    "model_xl": "openai/gpt-oss-safeguard-20b",              # change to real model if needed
    "model_l": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "model_m": "meta-llama/llama-guard-4-12b",
    "model_s": "groq/compound-mini",                         # fast & cheap
}

ROUTES = {
    "model_xl": "Safety analysis, policy sensitive, high risk content",
    "model_l": "Complex reasoning, system design, deep explanations",
    "model_m": "Cybersecurity, vulnerabilities, exploits, moderation",
    "model_s": "Casual chat, greetings, short or simple questions",
}

# Router model (small & fast)
router_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

ROUTE_EMBEDDINGS = {
    k: router_model.encode(v, normalize_embeddings=True)
    for k, v in ROUTES.items()
}

def route_prompt(prompt: str, threshold: float = 0.45) -> str:
    prompt_emb = router_model.encode(prompt, normalize_embeddings=True)
    scores = {
        k: cosine_similarity([prompt_emb], [v])[0][0]
        for k, v in ROUTE_EMBEDDINGS.items()
    }
    best_model, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score < threshold:
        return "model_s"  # fallback

    return best_model

# ────────────────────────────────────────────────
# LLM Call
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
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ────────────────────────────────────────────────
# WebSocket Events
# ────────────────────────────────────────────────
@socketio.on('message')
def handle_message(data):
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return

    # Show user message immediately
    emit('chat_message', {
        'role': 'user',
        'content': user_message
    })

    # Routing decision
    model_key = route_prompt(user_message)
    # emit('chat_message', {
    #     'role': 'system',
    #     'content': f"[routed to {model_key}]"           # removed bold
    # })

    # Get answer (streaming would be better, but simple version for now)
    try:
        answer = call_llm(model_key, user_message)
        emit('chat_message', {
            'role': 'assistant',
            'content': answer
        })
    except Exception as e:
        emit('chat_message', {
            'role': 'assistant',
            'content': f"Error occurred: {str(e)}"      # removed bold
        })

    emit('typing', {'status': False})  # optional

# ────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting Multi-LLM Web Chatbot...")
    print("Open: http://127.0.0.1:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)