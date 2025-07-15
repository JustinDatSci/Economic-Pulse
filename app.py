# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from fredapi import Fred
from openai import OpenAI
from dotenv import load_dotenv
import os

# This is a pre-written example of what the AI would generate.
SAMPLE_AI_SUMMARY = """
Based on the latest data, the economic climate shows mixed signals. While the unemployment rate remains low, indicating a strong labor market, inflation continues to be a concern, staying above the typical target. Recent GDP growth shows a modest expansion, but the elevated Federal Funds Rate suggests a continued policy focus on curbing inflation, which could temper future growth.
"""

# --- INITIAL SETUP ---
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")

# Initialize Fred API
try:
    fred = Fred(api_key=fred_api_key)
except Exception as e:
    st.error(f"Failed to initialize FRED API. Check your API key. Error: {e}")
    st.stop()

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_fred_data(series_id, series_name):
    """Fetches data for a given series ID from FRED."""
    try:
        df = fred.get_series(series_id).to_frame(name=series_name)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data for {series_name} ({series_id}). Error: {e}")
        return pd.DataFrame()

# Fetch all data
cpi_df = fetch_fred_data('CPIAUCSL', 'CPI')
unrate_df = fetch_fred_data('UNRATE', 'Unemployment Rate')
gdp_df = fetch_fred_data('A191RL1Q225SBEA', 'GDP Growth')
fed_funds_df = fetch_fred_data('DFF', 'Fed Funds Rate')
treasury_10y_df = fetch_fred_data('DGS10', '10Y Treasury Yield')

# In app.py, after the data fetching section
st.set_page_config(layout="wide", page_title="Economic Pulse Dashboard")
st.title("🇺🇸 Economic Pulse Dashboard")
st.markdown("A real-time snapshot of key U.S. economic indicators.")

# --- DATA PROCESSING ---
# Calculate Year-over-Year CPI change
cpi_df['YoY Growth (%)'] = cpi_df['CPI'].pct_change(12) * 100

# Get latest values
latest_cpi_growth = cpi_df['YoY Growth (%)'].iloc[-1]
latest_unrate = unrate_df['Unemployment Rate'].iloc[-1]
latest_gdp_growth = gdp_df['GDP Growth'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Inflation (CPI YoY)", f"{latest_cpi_growth:.2f}%")
col2.metric("Unemployment Rate", f"{latest_unrate:.2f}%")
col3.metric("GDP Growth (Quarterly)", f"{latest_gdp_growth:.2f}%")

def create_line_chart(df, title, y_axis_title, hover_name):
    """Creates a line chart using Plotly Express."""
    fig = px.line(df, x=df.index, y=df.columns[0], title=title, labels={'value': y_axis_title, 'index': 'Date'})
    fig.update_layout(hovermode="x unified")
    return fig

# Create two columns for the main charts
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(create_line_chart(cpi_df[['YoY Growth (%)']].dropna(), 'Inflation Rate (YoY % Change)', 'Percent', 'YoY Growth'), use_container_width=True)
    st.plotly_chart(create_line_chart(gdp_df, 'Real GDP Growth (Quarterly %)', 'Percent', 'GDP Growth'), use_container_width=True)
with chart_col2:
    st.plotly_chart(create_line_chart(unrate_df, 'Unemployment Rate (%)', 'Percent', 'Unemployment Rate'), use_container_width=True)
    st.plotly_chart(create_line_chart(fed_funds_df, 'Effective Federal Funds Rate (%)', 'Percent', 'Fed Funds Rate'), use_container_width=True)

# In app.py
st.subheader("🤖 AI Economic Analyst Summary")
openai_api_key = os.getenv("OPENAI_API_KEY")

if st.button("Generate AI Summary"):
    with st.spinner("The AI Analyst is thinking..."):
        # Display the pre-written sample summary
        st.info(SAMPLE_AI_SUMMARY)
        st.caption("Note: This is a sample AI summary to demonstrate functionality. Live API calls are disabled.")
