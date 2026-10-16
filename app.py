import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_manager import generate_synthetic_data, process_uploaded_csv
from src.categorizer import TAXONOMY
from src.insights import generate_insights
from src.utils import format_currency

# --- Config ---
st.set_page_config(page_title="SpendWise AI", page_icon="🏦", layout="wide")
CURRENCY = "BRL"

# --- CSS ---
st.markdown("""
<style>
    .stMetric {
        background-color: #F0F4F8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #003366;
    }
    .stMetric label {
        color: #003366 !important;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #003366;
    }
</style>
""", unsafe_allow_html=True)

# --- State ---
if 'df' not in st.session_state:
    st.session_state.df = generate_synthetic_data(50, CURRENCY)

# --- Helpers ---
def update_category(tx_id, new_category):
    idx = st.session_state.df.index[st.session_state.df['Transaction ID'] == tx_id].tolist()[0]
    st.session_state.df.at[idx, 'Final Category'] = new_category
    st.session_state.df.at[idx, 'Review Status'] = 'Corrected by User'

def confirm_category(tx_id):
    idx = st.session_state.df.index[st.session_state.df['Transaction ID'] == tx_id].tolist()[0]
    st.session_state.df.at[idx, 'Review Status'] = 'Confirmed by User'

def reset_category(tx_id):
    idx = st.session_state.df.index[st.session_state.df['Transaction ID'] == tx_id].tolist()[0]
    orig_cat = st.session_state.df.at[idx, 'Predicted Category']
    st.session_state.df.at[idx, 'Final Category'] = orig_cat
    
    conf = st.session_state.df.at[idx, 'Confidence']
    if conf < 0.60 or "UNKNOWN" in str(st.session_state.df.at[idx, 'Description']).upper():
        st.session_state.df.at[idx, 'Review Status'] = 'Needs Review'
    elif conf >= 0.8:
        st.session_state.df.at[idx, 'Review Status'] = 'High Confidence'
    else:
        st.session_state.df.at[idx, 'Review Status'] = 'Medium Confidence'

# --- Sidebar ---
st.sidebar.title("🏦 SpendWise AI")
st.sidebar.markdown("Intelligent Expense Categorization")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "Dashboard", 
    "Transactions", 
    "AI Review Center", 
    "Spending Insights", 
    "Import / Export", 
    "About & Privacy"
])

st.sidebar.divider()
if st.sidebar.button("🔄 Reset Demo Data", use_container_width=True):
    st.session_state.df = generate_synthetic_data(50, CURRENCY)
    st.rerun()

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("Dashboard Overview")
    
    df = st.session_state.df
    
    col1, col2, col3, col4 = st.columns(4)
    total_spend = df['Amount'].sum()
    tx_count = len(df)
    cat_count = df['Final Category'].nunique()
    needs_review = len(df[df['Review Status'] == 'Needs Review'])
    
    col1.metric("Total Spending", format_currency(total_spend, CURRENCY))
    col2.metric("Transactions", tx_count)
    col3.metric("Categories Used", cat_count)
    col4.metric("Needs Review", needs_review)
    
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Total Spending by Category")
        summary_df = df.groupby('Final Category')['Amount'].sum().reset_index()
        fig1 = px.bar(summary_df, x='Final Category', y='Amount', color_discrete_sequence=['#003366'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        st.subheader("Percentage of Spending")
        fig2 = px.pie(summary_df, values='Amount', names='Final Category', hole=0.4, color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.divider()
    
    col_table1, col_table2 = st.columns(2)
    
    with col_table1:
        st.subheader("Category Summary")
        st.dataframe(summary_df.sort_values('Amount', ascending=False), hide_index=True, use_container_width=True)
        
    with col_table2:
        st.subheader("Recent Transactions")
        recent_df = df.sort_values('Date', ascending=False).head(5)
        st.dataframe(recent_df[['Date', 'Description', 'Amount', 'Final Category']], hide_index=True, use_container_width=True)

# --- Page: Transactions ---
elif page == "Transactions":
    st.title("Transaction Management")
    
    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        cat_filter = col1.multiselect("Category", TAXONOMY)
        status_filter = col2.multiselect("Review Status", ["High Confidence", "Medium Confidence", "Low Confidence", "Needs Review", "Confirmed by User", "Corrected by User"])
        search = col3.text_input("Search Description")
        
    df_filtered = st.session_state.df.copy()
    if cat_filter:
        df_filtered = df_filtered[df_filtered['Final Category'].isin(cat_filter)]
    if status_filter:
        df_filtered = df_filtered[df_filtered['Review Status'].isin(status_filter)]
    if search:
        df_filtered = df_filtered[df_filtered['Description'].str.contains(search, case=False, na=False)]
        
    st.dataframe(
        df_filtered[['Transaction ID', 'Date', 'Description', 'Amount', 'Predicted Category', 'Confidence', 'Final Category', 'Review Status', 'Explanation']],
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()
    st.subheader("Human Correction Workflow")
    selected_tx_id = st.selectbox("Select Transaction ID to Edit", df_filtered['Transaction ID'].tolist())
    
    if selected_tx_id:
        tx = st.session_state.df[st.session_state.df['Transaction ID'] == selected_tx_id].iloc[0]
        
        st.info(f"**Original Description:** {tx['Description']} | **Amount:** {tx['Amount']} {tx['Currency']}")
        st.write(f"**AI Predicted Category:** {tx['Predicted Category']}")
        st.write(f"**Confidence:** {tx['Confidence']:.0%} ({tx['Method']})")
        st.write(f"**Explanation:** {tx['Explanation']}")
        
        new_cat = st.selectbox("Final Category", TAXONOMY, index=TAXONOMY.index(tx['Final Category']))
        
        col_btn1, col_btn2 = st.columns([1, 5])
        if col_btn1.button("Save Correction", type="primary"):
            update_category(selected_tx_id, new_cat)
            st.rerun()
        if col_btn2.button("Undo / Reset"):
            reset_category(selected_tx_id)
            st.rerun()

# --- Page: AI Review Center ---
elif page == "AI Review Center":
    st.title("AI Review Center")
    st.markdown("Review ambiguous transactions that the AI flagged for manual review.")
    
    review_df = st.session_state.df[st.session_state.df['Review Status'] == 'Needs Review']
    
    if review_df.empty:
        st.success("🎉 All transactions have been reviewed!")
    else:
        for _, tx in review_df.iterrows():
            with st.container():
                st.markdown(f"### {tx['Description']} - {tx['Amount']} {tx['Currency']}")
                st.write(f"**Predicted Category:** {tx['Predicted Category']} | **Confidence:** {tx['Confidence']:.0%}")
                st.warning(f"**Explanation:** {tx['Explanation']}")
                
                new_cat = st.selectbox("Select Appropriate Category", TAXONOMY, index=TAXONOMY.index(tx['Predicted Category']), key=f"sel_{tx['Transaction ID']}")
                
                col1, col2 = st.columns([1, 5])
                if col1.button("Confirm & Save", key=f"btn_{tx['Transaction ID']}", type="primary"):
                    if new_cat == tx['Predicted Category']:
                        confirm_category(tx['Transaction ID'])
                    else:
                        update_category(tx['Transaction ID'], new_cat)
                    st.rerun()
                st.divider()

# --- Page: Spending Insights ---
elif page == "Spending Insights":
    st.title("Spending Insights")
    st.markdown("Insights are based only on the loaded transactions. Do not infer income, financial health, or spending abnormalities without appropriate evidence.")
    
    insights = generate_insights(st.session_state.df)
    for insight in insights:
        st.info(f"💡 {insight}")

# --- Page: Import / Export ---
elif page == "Import / Export":
    st.title("Import / Export")
    
    st.subheader("Export Data")
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Categorized Data as CSV",
        data=csv,
        file_name='spendwise_export.csv',
        mime='text/csv',
    )
    
    st.divider()
    
    st.subheader("Import Data")
    st.markdown("Upload an anonymized CSV file containing transaction descriptions, dates, and amounts. Do not upload PII.")
    
    # Provide sample CSV
    sample_data = pd.DataFrame({
        "Date": ["2023-10-01", "2023-10-02"],
        "Description": ["UBER TRIP", "UNKNOWN ONLINE PAYMENT"],
        "Amount": [25.50, 100.00]
    })
    st.download_button(
        label="Download Sample CSV",
        data=sample_data.to_csv(index=False).encode('utf-8'),
        file_name='sample_transactions.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            processed_df = process_uploaded_csv(df_upload, CURRENCY)
            
            # Merge or replace
            if st.button("Replace Current Data with Uploaded Data"):
                st.session_state.df = processed_df
                st.success("Data successfully imported and categorized!")
                st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# --- Page: About & Privacy ---
elif page == "About & Privacy":
    st.title("About & Privacy")
    st.markdown("""
    ### SpendWise AI Prototype
    This application demonstrates an AI-powered approach to credit card expense categorization.
    
    #### 🔒 Privacy & Guardrails
    * **Synthetic Data Only:** All transactions displayed by default are synthetically generated.
    * **No PII:** No Personally Identifiable Information or real financial data is processed.
    * **Local Processing:** The AI classification runs entirely locally using Scikit-learn. No data is sent to external LLM APIs.
    * **Human-in-the-Loop:** All AI decisions are explainable and subject to human review and correction in the AI Review Center.
    * **Prototype Notice:** This application must only provide categorization and informational spending summaries. Do not use it to make lending, credit approval, fraud accusations, or other consequential financial decisions.
    """)
