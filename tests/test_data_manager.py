import pytest
import pandas as pd
from src.data_manager import process_uploaded_csv, generate_synthetic_data

def test_generate_synthetic_data():
    df = generate_synthetic_data(10)
    assert len(df) == 10
    assert "Transaction ID" in df.columns
    assert "Predicted Category" in df.columns
    assert "Review Status" in df.columns

def test_process_uploaded_csv_valid():
    df = pd.DataFrame({
        "Date": ["2023-10-01"],
        "Description": ["UBER TRIP"],
        "Amount": [25.50]
    })
    processed = process_uploaded_csv(df)
    assert len(processed) == 1
    assert processed.iloc[0]["Final Category"] == "Transportation"
    assert processed.iloc[0]["Review Status"] == "High Confidence"

def test_process_uploaded_csv_missing_cols():
    df = pd.DataFrame({
        "Date": ["2023-10-01"],
        "Amount": [25.50]
    })
    with pytest.raises(ValueError, match="Missing required column: Description"):
        process_uploaded_csv(df)

def test_process_uploaded_csv_invalid_amount():
    df = pd.DataFrame({
        "Date": ["2023-10-01"],
        "Description": ["UBER TRIP"],
        "Amount": ["invalid"]
    })
    processed = process_uploaded_csv(df)
    assert processed.iloc[0]["Amount"] == 0.0
