from flask import Flask, request, jsonify, send_from_directory
import os
import json
import threading
import requests
from groq import Groq
try:
    from model_code import load_model_and_data, get_btc_prediction, get_current_btc_price
except ImportError as e:
    print(f"Warning: Could not import model_code: {e}")
    # Fallback functions if model_code is not available
    def load_model_and_data():
        print("Model loading skipped - using fallback")
        return True
    
    def get_btc_prediction():
        return "BTC prediction model not loaded. Please check model_code.py"
    
    def get_current_btc_price():
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr")
            price = resp.json()['bitcoin']['inr']
            return f"Current Bitcoin price (INR): ₹{price:,.2f}"
        except:
            return "Unable to fetch current BTC price"

app = Flask(__name__, static_folder='static')

# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "c7e8585329cc9ec70e0c7533b6c644648cc0b8ae")

# Initialize Groq client
client = None
try:
    client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Groq client failed: {e}")

# Load model on startup
try:
    load_model_and_data()
    print("Model and data loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")

# ===== MAIN ROUTES =====

@app.route('/')
def index():
    """Serve the main chat interface"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        return f"""
        <html>
            <body>
                <h1>BTC Chatbot</h1>
                <p>Frontend not found. Error: {e}</p>
                <p>Try accessing /api/chat directly with POST requests.</p>
                <p>Example: POST /api/chat with JSON: {{"message": "hello"}}</p>
            </body>
        </html>
        """, 200

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "groq_client": "available" if client else "unavailable",
        "timestamp": "2025-08-16"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'reply': "No JSON data received"}), 400
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'reply': "Please type something!"})
        
        message_lower = message.lower()
        
        # Route to appropriate handler
        if any(word in message_lower for word in ['btc', 'bitcoin', 'crypto', 'predict', 'price']):
            if any(word in message_lower for word in ['current', 'live', 'now']):
                reply = get_current_btc_price()
            elif any(word in message_lower for word in ['predict', 'forecast', 'tomorrow', 'future']):
                reply = get_btc_prediction()
            elif 'help' in message_lower:
                reply = """I can help you with:
• Current BTC price: Ask "current bitcoin price"
• Price predictions: Ask "predict bitcoin price"
• General questions: Ask anything else!"""
            else:
                reply = "Ask me about current BTC price, price predictions, or any general questions!"
        else:
            # Use Groq for general questions
            reply = get_groq_response(message)
        
        return jsonify({'reply': reply})
        
    except Exception as e:
        return jsonify({'reply': f"Sorry, an error occurred: {str(e)[:200]}"}), 500

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Test endpoint for debugging"""
    if request.method == 'GET':
        return jsonify({
            "message": "Test endpoint working",
            "groq_status": "available" if client else "unavailable",
            "method": "GET"
        })
    else:
        data = request.get_json()
        return jsonify({
            "received_data": data,
            "groq_status": "available" if client else "unavailable",
            "method": "POST"
        })

# ===== HELPER FUNCTIONS =====

def get_groq_response(message):
    """Get response from Groq LLM"""
    if not client:
        return "AI service is currently unavailable. Please try again later."
    
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
                },
                {
                    "role": "user", 
                    "content": message
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return "I'm experiencing high traffic. Please try again in a moment."
        elif "api_key" in error_msg.lower():
            return "Authentication issue with AI service. Please contact support."
        else:
            return f"AI service error: {error_msg[:100]}... Please try again."

def google_search(query):
    """Search using Serper API"""
    if not SERPER_API_KEY or SERPER_API_KEY == "your_serper_key_here":
        return "Search service not configured"
    
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={"q": query},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("organic"):
            return "No search results found"
        
        results = []
        for item in data["organic"][:3]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No description')
            results.append(f"• {title}\n  {snippet}")
        
        return "\n\n".join(results)
        
    except Exception as e:
        return f"Search service error: {str(e)[:100]}"

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
        "message": "Please check the HTTP method for this endpoint"
    }), 405

# ===== CORS SUPPORT (if needed) =====

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
