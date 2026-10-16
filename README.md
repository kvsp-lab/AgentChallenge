# SpendWise AI

SpendWise AI is a professional, AI-powered credit card expense categorization agent built with Python and Streamlit. It automatically categorizes transaction descriptions, explains uncertain predictions, allows users to correct categories, and summarizes spending by category.

## Features
- **Dashboard**: High-level KPIs, spending charts, and recent transactions.
- **Transaction Management**: Interactive table with filtering, sorting, and a human correction workflow.
- **AI Review Center**: Dedicated section for reviewing ambiguous transactions flagged by the AI.
- **Spending Insights**: Dynamically generated text summaries based on the loaded dataset.
- **Import / Export**: Upload anonymized CSVs or download categorized data.

## Privacy & Guardrails
- **Synthetic Data Only**: Uses synthetic demo data by default.
- **No PII**: Does not process or store Personally Identifiable Information (PII).
- **Local Processing**: The AI classification runs entirely locally using Scikit-learn. No data is sent to external LLM APIs.
- **Human-in-the-Loop**: All AI decisions are explainable and subject to human review and correction.
- **Prototype Notice**: This application is a prototype for categorization and informational summaries only. Do not use it for consequential financial decisions.

## Setup Instructions

### 1. Create and activate a virtual environment
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

### 4. Run automated tests
```bash
pytest
```

## Architecture
- `app.py`: Main Streamlit application and UI routing.
- `src/categorizer.py`: TF-IDF and Logistic Regression model combined with rule-based overrides.
- `src/data_manager.py`: Synthetic data generation and CSV processing logic.
- `src/insights.py`: Dynamic text summary generation.
- `src/utils.py`: Formatting and styling helpers.
- `tests/`: Pytest suite for the categorizer and data manager.

## Known Limitations
- The AI model is trained on a small, synthetic seed dataset. In a production environment, it would require a much larger, diverse dataset for high accuracy across all real-world merchants.
- CSV Import currently keeps data in session memory only; it does not persist to a database.