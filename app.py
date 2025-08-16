from flask import Flask, request, jsonify, send_from_directory
import os
import json
import threading
import requests
from groq import Groq
from model_code import load_model_and_data, get_btc_prediction, get_current_btc_price

app = Flask(__name__, static_folder='static')

# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "c7e8585329cc9ec70e0c7533b6c644648cc0b8ae")

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    client = None
    print(f"Groq client failed: {e}")

# Load model on startup
load_model_and_data()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').strip().lower()
    
    if not message:
        return jsonify({'reply': "Please type something!"})
    
    # Route to appropriate handler
    if any(word in message for word in ['btc', 'bitcoin', 'crypto', 'predict', 'price']):
        if 'current' in message or 'live' in message:
            reply = get_current_btc_price()
        elif 'predict' in message or 'forecast' in message:
            reply = get_btc_prediction()
        else:
            reply = "Ask me about current BTC price or price predictions!"
    else:
        # Use Groq for general questions
        reply = get_groq_response(message)
    
    return jsonify({'reply': reply})

def get_groq_response(message):
    """Get response from Groq LLM"""
    if not client:
        return "AI service unavailable"
    
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": message}],
            max_tokens=1000,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)[:200]}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
