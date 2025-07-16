# app.py
# Standard library imports (built into Python)
import os
from datetime import datetime, timedelta

# Third-party library imports (installed with pip)
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from fredapi import Fred
from openai import OpenAI

# --- ANIMATION ---
st.markdown("""
<style>
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* A more specific and forceful rule to prevent Streamlit from overriding it */
h1 .pulse-text {
    animation: pulse 2s infinite !important;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# --- INITIAL SETUP ---
load_dotenv()
fred_api_key = st.secrets["FRED_API_KEY"]

try:
    fred = Fred(api_key=fred_api_key)
except Exception as e:
    st.error(f"Failed to initialize FRED API. Check your API key. Error: {e}")
    st.stop()

# --- DATA & AI FUNCTIONS ---
ECONOMIC_SERIES = {
    "Inflation (CPI YoY)": "CPIAUCSL",
    "Unemployment Rate": "UNRATE",
    "GDP Growth (Quarterly)": "A191RL1Q225SBEA",
    "Effective Federal Funds Rate": "DFF",
    "10Y Treasury Yield": "DGS10",
}

@st.cache_data(ttl=3600, show_spinner="Fetching economic data...")
def load_all_data(series_dict, start_date=None):
    """Fetches all data series defined in a dictionary."""
    data = {}
    for name, series_id in series_dict.items():
        try:
            #pass start_date to the API call here
            df = fred.get_series(series_id, observation_start=start_date).to_frame(name=name)
            data[name] = df
        except Exception as e:
            st.error(f"Failed to fetch {name} ({series_id}). Error: {e}")
            data[name] = pd.DataFrame()
    return data

@st.cache_data(ttl=3600)
def get_ai_summary(latest_data_string):
    """Generates an economic summary using the OpenAI API."""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = f"""
    Analyze the current U.S. economic climate based on the following data points:
    {latest_data_string}
    Provide a concise, neutral summary of what these indicators suggest.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- APP LAYOUT ---
st.markdown(
    '<h1>🇺🇸 Economic <span class="pulse-text">Pulse</span> Dashboard</h1>',
    unsafe_allow_html=True
)
st.markdown("A real-time snapshot of key U.S. economic indicators.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Dashboard Controls")

st.sidebar.subheader("Data Selection")

# Create a list of all indicator names
all_indicator_names = list(ECONOMIC_SERIES.keys())

# Horizontal layout with one column per indicator
cols = st.sidebar.columns(len(all_indicator_names))

# Checkbox in each column and collect the selected ones
selected_indicators = []
for i, indicator_name in enumerate(all_indicator_names):
    # Default to the first 4 being selected
    is_selected = cols[i].checkbox(indicator_name, value=(i < 4))
    if is_selected:
        selected_indicators.append(indicator_name)

# Time period selector
time_period = st.sidebar.selectbox(
    "Select Time Period",
    options=['1Y', '3Y', '5Y', '10Y', 'All Time'],
    index=2  # Default to '5Y'
)

st.sidebar.divider()

# Button to clear the cache
if st.sidebar.button("Clear Cache & Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

st.sidebar.markdown(
    """
    **About**
    This dashboard provides a real-time snapshot of key U.S. economic indicators.
    Data is sourced from the [FRED API](https://fred.stlouisfed.org/docs/api/fred/).
    """
)

# --- DATA LOADING ---
end_date = datetime.now()

# Special handling for YoY calculation
# If user wants 1Y of inflation, we need 2Y of raw data to calculate it
if time_period == '1Y' and "Inflation (CPI YoY)" in selected_indicators:
    start_date = end_date - timedelta(days=2 * 365)
    st.sidebar.info("Note: Fetching 2 years of data to calculate 1-year inflation change.", icon="ℹ️")
elif time_period == 'All Time':
    start_date = None
else:
    years = int(time_period.replace('Y', ''))
    start_date = end_date - timedelta(days=years * 365)

# Filter the series dict based on user selection
series_to_load = {name: sid for name, sid in ECONOMIC_SERIES.items() if name in selected_indicators}

# Load the data for the calculated time period
all_data = load_all_data(series_to_load, start_date=start_date)

# --- DYNAMIC METRIC DISPLAY ---
if not selected_indicators:
    st.warning("Please select at least one indicator from the sidebar.")
else:
    latest_values = {}
    cols = st.columns(len(selected_indicators))
    for i, indicator_name in enumerate(selected_indicators):
        with cols[i]:
            df = all_data[indicator_name].copy()
            if indicator_name == "Inflation (CPI YoY)":
                df['YoY Growth (%)'] = df[indicator_name].pct_change(12) * 100
                latest_value = df['YoY Growth (%)'].iloc[-1]
                previous_value = df['YoY Growth (%)'].iloc[-2]
                label = "Inflation (CPI YoY)"
            else:
                latest_value = df[indicator_name].iloc[-1]
                previous_value = df[indicator_name].iloc[-2]
                label = indicator_name
            
            latest_values[label] = latest_value
            delta = latest_value - previous_value
            st.metric(label=label, value=f"{latest_value:.2f}%", delta=f"{delta:.2f}%")

# --- DYNAMIC CHARTING SECTION ---
def create_line_chart(df, title, y_axis_title):
    """Creates a line chart using Plotly Express."""
    fig = px.line(df, x=df.index, y=df.columns[0], title=title, labels={'value': y_axis_title, 'index': 'Date'})
    fig.update_layout(hovermode="x unified")
    return fig

if selected_indicators:
    st.markdown("---")
    st.subheader("Indicator Charts")
    chart_col1, chart_col2 = st.columns(2)
    for i, indicator_name in enumerate(selected_indicators):
        df_chart = all_data[indicator_name].copy()
        if indicator_name == "Inflation (CPI YoY)":
            df_chart['YoY Growth (%)'] = df_chart[indicator_name].pct_change(12) * 100
            chart_df_to_plot = df_chart[['YoY Growth (%)']].dropna()
        else:
            chart_df_to_plot = df_chart
        
        if i % 2 == 0:
            with chart_col1:
                st.plotly_chart(create_line_chart(chart_df_to_plot, indicator_name, 'Percent'), use_container_width=True)
        else:
            with chart_col2:
                st.plotly_chart(create_line_chart(chart_df_to_plot, indicator_name, 'Percent'), use_container_width=True)

# --- 3. DYNAMIC AI SUMMARY SECTION ---
st.markdown("---")
st.subheader("🤖 AI Economic Analyst Summary")
if st.button("Generate Live AI Summary"):
    if not selected_indicators:
        st.warning("Please select indicators to generate a summary.")
    else:
        with st.spinner("The AI Analyst is thinking..."):
            # Build the prompt string dynamically from the selected indicators
            prompt_data_parts = [f"- {name}: {value:.2f}%" for name, value in latest_values.items()]
            latest_data_str = "\n".join(prompt_data_parts)
            summary = get_ai_summary(latest_data_str)
            st.info(summary)

# --- FOOTER ---
st.markdown("---")
st.markdown("Source Code: [GitHub](https://github.com/JustinDatSci/Economic-Pulse)")