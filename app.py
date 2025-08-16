from flask import Flask, request, jsonify, send_from_directory
import os
import threading
import requests
from groq import Groq

# Attempt to import your model code, otherwise use fallbacks
try:
    from model_code import load_model_and_data, get_btc_prediction, get_current_btc_price
except ImportError as e:
    print(f"Warning: Could not import model_code: {e}")

    def load_model_and_data():
        print("Model loading skipped – using fallback")
        return True

    def get_btc_prediction():
        return "BTC prediction model not loaded. Please check model_code.py"

    def get_current_btc_price():
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr"
            )
            price = resp.json()['bitcoin']['inr']
            return f"Current Bitcoin price (INR): ₹{price:,.2f}"
        except:
            return "Unable to fetch current BTC price"

# Initialize Flask app, serving static files from "static" folder at root path
app = Flask(__name__, static_folder='static', static_url_path='')

# Load environment variables for API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_DEFAULT_GROQ_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "YOUR_DEFAULT_SERPER_KEY")

# Initialize Groq client
client = None
try:
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Groq client failed: {e}")

# Load your ML model and data at startup
try:
    load_model_and_data()
    print("Model and data loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")

# ===== MAIN ROUTES =====

@app.route('/')
def index():
    """
    Serve the main chat interface HTML from static/index.html
    """
    return app.send_static_file('index.html')

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "groq_client": "available" if client else "unavailable",
        "timestamp": "2025-08-17"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'reply': "No JSON data received"}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({'reply': "Please type something!"})

        m = message.lower()
        # Route finance queries to your model
        if any(k in m for k in ['btc', 'bitcoin', 'crypto', 'predict', 'price']):
            if any(k in m for k in ['current', 'live', 'now']):
                reply = get_current_btc_price()
            elif any(k in m for k in ['predict', 'forecast', 'tomorrow', 'future']):
                reply = get_btc_prediction()
            elif 'help' in m:
                reply = (
                    "I can help you with:\n"
                    "• Current BTC price: Ask 'current bitcoin price'\n"
                    "• Price predictions: Ask 'predict bitcoin price'\n"
                    "• General questions: Ask me anything else!"
                )
            else:
                reply = "Ask me about current BTC price, predictions, or general questions!"
        else:
            # Fall back to Groq for other queries
            reply = get_groq_response(message)

        return jsonify({'reply': reply})

    except Exception as e:
        return jsonify({'reply': f"Sorry, an error occurred: {str(e)[:200]}"}), 500

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """
    Test endpoint for debugging
    """
    if request.method == 'GET':
        return jsonify({
            "message": "Test endpoint working",
            "groq_status": "available" if client else "unavailable",
            "method": "GET"
        })
    data = request.get_json()
    return jsonify({
        "received_data": data,
        "groq_status": "available" if client else "unavailable",
        "method": "POST"
    })

# ===== HELPER FUNCTIONS =====

def get_groq_response(message: str) -> str:
    """
    Query Groq LLM for general questions
    """
    if not client:
        return "AI service is currently unavailable. Please try again later."
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        err = str(e).lower()
        if "rate_limit" in err:
            return "I'm experiencing high traffic. Please try again shortly."
        if "api_key" in err:
            return "Authentication issue with AI service. Please contact support."
        return f"AI service error: {str(e)[:100]}... Please try again."

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET / - Main chat interface",
            "POST /api/chat - Chat with bot",
            "GET /health - Health check",
            "GET|POST /api/test - Test endpoint"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on our end"
    }), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "Check the HTTP method for this endpoint"
    }), 405

# ===== CORS SUPPORT =====

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    print(f"Groq client: {'Available' if client else 'Unavailable'}")
    app.run(host="0.0.0.0", port=port, debug=False)
