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

def generate_synthetic_data(n=50, currency="BRL"):
    categorizer = ExpenseCategorizer()
    data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for _ in range(n):
        desc = random.choice(DESCRIPTIONS)
        amount = round(random.uniform(15.0, 1500.0), 2)
        date = start_date + timedelta(days=random.randint(0, 30))
        
        res = categorizer.categorize(desc)
        
        if res["needs_review"]:
            status = "Needs Review"
        elif res["confidence"] >= 0.8:
            status = "High Confidence"
        elif res["confidence"] >= 0.6:
            status = "Medium Confidence"
        else:
            status = "Low Confidence"
            
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
            "Method": res["method"],
            "Alternatives": res["alternatives"]
        }
        data.append(tx)
        
    return pd.DataFrame(data)
