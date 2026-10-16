import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TAXONOMY = [
    "Groceries", "Dining & Restaurants", "Transportation", "Fuel", 
    "Shopping", "Entertainment & Subscriptions", "Travel & Accommodation", 
    "Utilities & Bills", "Healthcare & Pharmacy", "Education", 
    "Fitness & Wellness", "Other / Needs Review"
]

TRAIN_DATA = [
    ("SUPERMARKET PURCHASE", "Groceries"),
    ("WHOLE FOODS", "Groceries"),
    ("TRADER JOES", "Groceries"),
    ("RESTAURANT PAYMENT", "Dining & Restaurants"),
    ("UBER EATS", "Dining & Restaurants"),
    ("CHIPOTLE", "Dining & Restaurants"),
    ("UBER TRIP", "Transportation"),
    ("LYFT RIDE", "Transportation"),
    ("SHELL FUEL STATION", "Fuel"),
    ("EXXON MOBIL", "Fuel"),
    ("AMAZON MARKETPLACE", "Shopping"),
    ("TARGET STORE", "Shopping"),
    ("WALMART SUPERCENTER", "Shopping"),
    ("NETFLIX SUBSCRIPTION", "Entertainment & Subscriptions"),
    ("SPOTIFY", "Entertainment & Subscriptions"),
    ("HULU", "Entertainment & Subscriptions"),
    ("HOTEL BOOKING", "Travel & Accommodation"),
    ("AIRBNB", "Travel & Accommodation"),
    ("DELTA AIRLINES", "Travel & Accommodation"),
    ("ELECTRICITY BILL", "Utilities & Bills"),
    ("WATER UTILITY", "Utilities & Bills"),
    ("CON EDISON", "Utilities & Bills"),
    ("PHARMACY PURCHASE", "Healthcare & Pharmacy"),
    ("CVS PHARMACY", "Healthcare & Pharmacy"),
    ("ONLINE COURSE", "Education"),
    ("COURSERA", "Education"),
    ("GYM MEMBERSHIP", "Fitness & Wellness"),
    ("PLANET FITNESS", "Fitness & Wellness"),
    ("UNKNOWN ONLINE PAYMENT", "Other / Needs Review"),
    ("MISC CHARGE", "Other / Needs Review")
]

RULES = {
    "uber": "Transportation",
    "lyft": "Transportation",
    "netflix": "Entertainment & Subscriptions",
    "spotify": "Entertainment & Subscriptions",
    "shell": "Fuel",
    "exxon": "Fuel",
    "cvs": "Healthcare & Pharmacy",
    "walgreens": "Healthcare & Pharmacy",
    "amazon": "Shopping",
    "target": "Shopping",
    "walmart": "Shopping",
    "whole foods": "Groceries",
    "trader joe": "Groceries"
}

class ExpenseCategorizer:
    def __init__(self, threshold=0.60):
        self.threshold = threshold
        self.pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
        X, y = zip(*TRAIN_DATA)
        self.pipeline.fit(X, y)
        self.classes = self.pipeline.classes_

    def normalize_text(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def categorize(self, description):
        norm_desc = self.normalize_text(description)
        
        if not norm_desc:
            return {
                "predicted_category": "Other / Needs Review",
                "confidence": 0.0,
                "explanation": "Empty or invalid description provided.",
                "method": "Fallback",
                "needs_review": True
            }

        # 1. Rule-based check
        for keyword, category in RULES.items():
            if keyword in norm_desc:
                return {
                    "predicted_category": category,
                    "confidence": 0.95,
                    "explanation": f"The transaction description matches a known keyword pattern ('{keyword}').",
                    "method": "Rule-based",
                    "needs_review": False
                }

        # 2. Model-based check
        probs = self.pipeline.predict_proba([norm_desc])[0]
        top_idx = np.argmax(probs)
        confidence = float(probs[top_idx])
        category = self.classes[top_idx]
        
        needs_review = confidence < self.threshold

        if "unknown" in norm_desc or category == "Other / Needs Review":
            explanation = "The available description does not provide enough information to determine the purchase category."
            needs_review = True
            category = "Other / Needs Review"
        elif needs_review:
            explanation = "The transaction description is too generic to identify the purchase type reliably. The current prediction is uncertain. Please review the transaction and select the appropriate category."
        else:
            explanation = f"Model identified patterns consistent with {category}."

        return {
            "predicted_category": category,
            "confidence": confidence,
            "explanation": explanation,
            "method": "Model",
            "needs_review": needs_review
        }
