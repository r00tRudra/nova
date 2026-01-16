# Nova Chat - Multi-LLM Router Chatbot

A beautiful, minimalistic web-based chatbot that intelligently routes user prompts to different Groq models based on the type of question.

## Features

- Smart Model Routing — automatically chooses the best model based on prompt content  
- Supports 4 different Grok models with specialized roles
- Modern dark-themed chat UI with auto-growing input
- Real-time responses via WebSocket (Socket.IO)
- Hover toolbars, copy/edit/regenerate/love/dislike buttons (frontend ready)
- Generation time display
- Scroll-to-bottom button

## Currently Supported Models & Routing Logic

| Key     | Model ID                                      | Purpose / Specialty                              |
|---------|-----------------------------------------------|--------------------------------------------------|
| model_xl| openai/gpt-oss-safeguard-20b                  | Safety analysis, policy-sensitive content        |
| model_l | meta-llama/llama-4-maverick-17b-128e-instruct | Complex reasoning, system design, deep answers   |
| model_m | meta-llama/llama-guard-4-12b                  | Cybersecurity, exploits, moderation              |
| model_s | groq/compound-mini                            | Casual chat, greetings, simple & fast questions  |

Routing uses sentence-transformers + cosine similarity (very lightweight)

## Tech Stack

- Backend: Flask + Flask-SocketIO
- LLM: Groq API
- Embedding/Routing: sentence-transformers + scikit-learn
- Frontend: Pure HTML/CSS + JavaScript + Socket.IO
- Environment: python-dotenv

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/yourusername/nova-chat.git
cd nova-chat
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create .env file and add your key
```bash
GROQ_API_KEY=gq_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

4. Run the app
```bash
python main.py
```

5. Open in browser → http://127.0.0.1:5000


## Project Status (Jan 2026)

- Basic routing + chat working
- Nice dark UI with hover controls
- Missing: real user authentication, chat history persistence, rate limiting

