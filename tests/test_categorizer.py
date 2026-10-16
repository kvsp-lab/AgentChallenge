import pytest
from src.categorizer import ExpenseCategorizer

@pytest.fixture
def categorizer():
    return ExpenseCategorizer(threshold=0.60)

def test_known_transaction(categorizer):
    res = categorizer.categorize("UBER TRIP")
    assert res["predicted_category"] == "Transportation"
    assert res["method"] == "Rule-based"
    assert res["needs_review"] is False

def test_ambiguous_transaction(categorizer):
    res = categorizer.categorize("UNKNOWN ONLINE PAYMENT")
    assert res["predicted_category"] == "Other / Needs Review"
    assert res["needs_review"] is True
    assert "does not provide enough information" in res["explanation"]

def test_empty_description(categorizer):
    res = categorizer.categorize("")
    assert res["predicted_category"] == "Other / Needs Review"
    assert res["needs_review"] is True
    assert res["confidence"] == 0.0

def test_missing_description(categorizer):
    res = categorizer.categorize(None)
    assert res["predicted_category"] == "Other / Needs Review"
    assert res["needs_review"] is True
    assert res["confidence"] == 0.0

def test_model_prediction(categorizer):
    res = categorizer.categorize("SUPERMARKET PURCHASE")
    assert res["predicted_category"] == "Groceries"
    assert res["method"] == "Model"
