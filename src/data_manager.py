import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
from src.categorizer import ExpenseCategorizer

DESCRIPTIONS = [
    "NETFLIX SUBSCRIPTION", "UBER TRIP", "SUPERMARKET PURCHASE", 
    "SHELL FUEL STATION", "AMAZON MARKETPLACE", "RESTAURANT PAYMENT", 
    "PHARMACY PURCHASE", "ELECTRICITY BILL", "HOTEL BOOKING", 
    "GYM MEMBERSHIP", "ONLINE COURSE", "UNKNOWN ONLINE PAYMENT",
    "STARBUCKS COFFEE", "APPLE.COM/BILL", "DELTA AIRLINES",
    "WALMART SUPERCENTER", "CVS PHARMACY", "CON EDISON",
    "HULU", "LYFT RIDE", "CHIPOTLE", "TRADER JOES", "HOME DEPOT",
    "WHOLE FOODS", "SPOTIFY", "AIRBNB", "PLANET FITNESS", "COURSERA",
    "LOCAL BAKER", "GENERIC STORE 123", "PAYPAL *TRANSFER"
]

def get_status_label(confidence, needs_review):
    if needs_review:
        return "Needs Review"
    if confidence >= 0.8:
        return "High Confidence"
    if confidence >= 0.6:
        return "Medium Confidence"
    return "Low Confidence"

def generate_synthetic_data(n=50, currency="BRL"):
    categorizer = ExpenseCategorizer()
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for _ in range(n):
        desc = random.choice(DESCRIPTIONS)
        amount = round(random.uniform(15.0, 1500.0), 2)
        date = start_date + timedelta(days=random.randint(0, 30))
        
        res = categorizer.categorize(desc)
        status = get_status_label(res["confidence"], res["needs_review"])
            
        tx = {
            "Transaction ID": str(uuid.uuid4())[:8].upper(),
            "Date": date.strftime("%Y-%m-%d"),
            "Description": desc,
            "Amount": amount,
            "Currency": currency,
            "Predicted Category": res["predicted_category"],
            "Confidence": res["confidence"],
            "Final Category": res["predicted_category"],
            "Review Status": status,
            "Explanation": res["explanation"],
            "Method": res["method"]
        }
        data.append(tx)
        
    return pd.DataFrame(data)

def process_uploaded_csv(df, currency="BRL"):
    categorizer = ExpenseCategorizer()
    
    # Validate required columns
    required_cols = ["Date", "Description", "Amount"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Ensure Transaction ID exists
    if "Transaction ID" not in df.columns:
        df["Transaction ID"] = [str(uuid.uuid4())[:8].upper() for _ in range(len(df))]
        
    # Ensure Currency exists
    if "Currency" not in df.columns:
        df["Currency"] = currency
        
    # Process rows
    processed_data = []
    for _, row in df.iterrows():
        desc = str(row.get("Description", ""))
        try:
            amount = float(row.get("Amount", 0.0))
        except ValueError:
            amount = 0.0
            
        res = categorizer.categorize(desc)
        status = get_status_label(res["confidence"], res["needs_review"])
        
        tx = {
            "Transaction ID": str(row["Transaction ID"]),
            "Date": str(row["Date"]),
            "Description": desc,
            "Amount": amount,
            "Currency": str(row["Currency"]),
            "Predicted Category": res["predicted_category"],
            "Confidence": res["confidence"],
            "Final Category": res["predicted_category"],
            "Review Status": status,
            "Explanation": res["explanation"],
            "Method": res["method"]
        }
        processed_data.append(tx)
        
    return pd.DataFrame(processed_data)
