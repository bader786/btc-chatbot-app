import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

trading = pd.read_csv("BTC_data_5years_cleaned.csv")

trading.hist(bins=50,figsize=(20,15))


train_set,test_set = train_test_split(trading,test_size=0.2,random_state=42)


trading_numeric = trading.select_dtypes(include=["number"])
corr_matrix = trading_numeric.corr()
corr_matrix["Low"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ["Low","High","Open","Close","Volume"]
scatter_matrix(trading[attributes],figsize=(12,8))
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


trading['Close_next'] = trading['Close'].shift(-1)
trading = trading[:-1]
trading['Close_mean_3'] = trading['Close'].rolling(window=3).mean().shift(1)
trading['Volume_mean_3'] = trading['Volume'].rolling(window=3).mean().shift(1)
trading = trading.dropna()


features = ["Open", "High", "Low", "Volume", "Close_mean_3", "Volume_mean_3"]
X = trading[features]
y = trading['Close_next']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


last_row = trading.iloc[-1]
last_features = last_row[features].values.reshape(1, -1)


all_preds = np.array([tree.predict(last_features)[0] for tree in model.estimators_])

mean_pred = all_preds.mean()
std_pred = all_preds.std()


lower = mean_pred - std_pred
upper = mean_pred + std_pred

print(f"Predicted next day close BTC-INR price range: ₹{lower:,.2f} to ₹{upper:,.2f} (mean: ₹{mean_pred:,.2f})")

latest_close = last_row["Close"]
if mean_pred > latest_close:
    print("Model advice: HOLD or BUY, price expected to rise (within predicted range).")
else:
    print("Model advice: SELL, price expected to fall (within predicted range).")
import sys
import requests
import json
import datetime
import pytz
import os
import time
import threading
from typing import Dict, List, Optional
from functools import lru_cache
from groq import Groq

CONFIG = {
    'max_tokens': 2048,
    'temperature': 0.6,
    'top_p': 0.95,
    'search_results': 3,
    'cache_size': 100,
    'min_response_chars': 10,
    'response_timeout': 10,
    'retry_attempts': 2
}


GROQ_API_KEY = "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"
SERPER_API_KEY = "c7e8585329cc9ec70e0c7533b6c644648cc0b8ae"

client = None
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Groq client initialization failed: {e}")

DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)
CHAT_LOG_PATH = os.path.join(DATA_DIR, "ChatLog.json")

def safe_json_load(path: str) -> List[Dict]:
    encodings = ['utf-8', 'utf-16', 'latin-1']
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeError, json.JSONDecodeError, FileNotFoundError):
            continue
    return []

def safe_json_dump(data: List[Dict], path: str) -> None:
    try:
        temp_path = f"{path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except Exception as e:
        print(f"Failed to save chat history: {e}")

@lru_cache(maxsize=32)
def get_cached_time(location: str) -> str:
    tz = pytz.timezone("Asia/Kolkata" if "india" in location.lower() else "UTC")
    now = datetime.datetime.now(tz)
    return (f"The current time in {location.title()} is "
            f"{now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}")

def google_search(query: str) -> str:
    if not SERPER_API_KEY:
        return "Search service is currently unavailable"
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("organic"):
            return "No relevant results found"
        return "\n\n".join(
            f"• {item.get('title', 'No title')}\n"
            f"  {item.get('link', 'No link')}\n"
            f"  {item.get('snippet', 'No description available')}"
            for item in data["organic"][:CONFIG['search_results']]
        )
    except Exception as e:
        return f"Search encountered an issue: {str(e)[:150]}"

def get_formatted_context() -> str:
    now = datetime.datetime.now()
    return (f"Current Date: {now.strftime('%A, %B %d, %Y')}\n"
            f"Current Time: {now.strftime('%I:%M %p %Z')}")

def ensure_complete_response(text: str) -> str:
    text = text.strip()
    if not text:
        return "I couldn't generate a response. Please try again."
    if text[-1] not in {'.', '!', '?', ':', ';'}:
        text += '.'
    if len(text) < CONFIG['min_response_chars']:
        text += " Please let me know if you need more details."
    return text

def generate_system_prompts() -> List[Dict]:
    assistant_name = "Assistant"
    return [
        {
            "role": "system",
            "content": (
                f"You are {assistant_name}, an AI assistant that provides "
                "detailed, complete responses. Your answers should:\n"
                "- Be at least 3-5 sentences long\n"
                "- Fully address the user's question\n"
                "- Include relevant details when available\n"
                "- End with proper punctuation"
            )
        },
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": (
            "Hello! I'm here to help you with any questions or tasks you have. "
            "I'll always provide complete, thoughtful responses to ensure you "
            "get the information you need. How can I assist you today?"
        )}
    ]

def get_complete_completion(messages: List[Dict]) -> Optional[str]:
    for attempt in range(CONFIG['retry_attempts']):
        try:
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=CONFIG['max_tokens'],
                temperature=CONFIG['temperature'],
                top_p=CONFIG['top_p'],
                stream=False
            )
            response = completion.choices[0].message.content.strip()
            if len(response) >= CONFIG['min_response_chars']:
                return response
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            print(f"Completion attempt {attempt + 1} failed: {e}")
    return None

def RealtimeSearchEngine(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "time" in prompt_lower and ("hyderabad" in prompt_lower or "india" in prompt_lower):
        return get_cached_time('hyderabad')
    try:
        messages = safe_json_load(CHAT_LOG_PATH)
        search_results = google_search(prompt)
        current_context = get_formatted_context()
        context = (
            f"User Query: {prompt}\n\n"
            f"Search Results:\n{search_results}\n\n"
            f"Context:\n{current_context}"
        )
        system_messages = generate_system_prompts()
        user_messages = messages[-5:] + [{"role": "user", "content": context}]
        answer = get_complete_completion(system_messages + user_messages)
        if not answer:
            return "I'm having trouble generating a complete response. Please try again."
        answer = ensure_complete_response(answer)
        updated_messages = messages + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        threading.Thread(
            target=safe_json_dump,
            args=(updated_messages[-CONFIG['cache_size']:], CHAT_LOG_PATH),
            daemon=True
        ).start()
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)[:200]}"

if __name__ == "__main__":
    print("Assistant ready. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            start_time = time.time()
            response = RealtimeSearchEngine(prompt)
            elapsed = time.time() - start_time
            print(f"\nAssistant ({elapsed:.2f}s):")
            print(response)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nSystem error: {e}")
