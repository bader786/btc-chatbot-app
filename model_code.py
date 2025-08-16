import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Global variables for model and data
model = None
trading = None
features = ["Open", "High", "Low", "Volume", "Close_mean_3", "Volume_mean_3"]

def load_model_and_data():
    """Load the trained model and prepare data"""
    global model, trading
    
    # Load data
    try:
        trading = pd.read_csv("Data/BTC_data_5years_cleaned.csv")
    except:
        # Fallback if CSV not available - you'll need to handle this
        trading = pd.DataFrame()  # Empty fallback
        return False
    
    # Feature engineering
    trading['Close_next'] = trading['Close'].shift(-1)
    trading = trading[:-1]
    trading['Close_mean_3'] = trading['Close'].rolling(window=3).mean().shift(1)
    trading['Volume_mean_3'] = trading['Volume'].rolling(window=3).mean().shift(1)
    trading = trading.dropna()
    
    # Try to load existing model, else train new one
    model_path = "Data/btc_rf_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Train model
        X = trading[features]
        y = trading['Close_next']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs("Data", exist_ok=True)
        joblib.dump(model, model_path)
    
    return True

def get_btc_prediction():
    """Get BTC price prediction range"""
    if model is None or trading.empty:
        return "Model not loaded"
    
    last_row = trading.iloc[-1]
    last_features = last_row[features].values.reshape(1, -1)
    
    # Get predictions from all trees
    all_preds = np.array([tree.predict(last_features)[0] for tree in model.estimators_])
    
    mean_pred = all_preds.mean()
    std_pred = all_preds.std()
    
    lower = mean_pred - std_pred
    upper = mean_pred + std_pred
    
    latest_close = last_row["Close"]
    advice = "HOLD or BUY" if mean_pred > latest_close else "SELL"
    
    return f"Predicted BTC-INR price range: ₹{lower:,.2f} to ₹{upper:,.2f} (mean: ₹{mean_pred:,.2f}). Advice: {advice}"

def get_current_btc_price():
    """Get live BTC price"""
    import requests
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr")
        price = resp.json()['bitcoin']['inr']
        return f"Current Bitcoin price (INR): ₹{price:,.2f}"
    except:
        return "Unable to fetch current price"
