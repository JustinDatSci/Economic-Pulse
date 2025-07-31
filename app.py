import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Economic Pulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_comprehensive_data():
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
    
    data = []
    indicators = {
        'UNRATE': {'name': 'Unemployment Rate', 'unit': '%', 'base': 4.0},
        'FEDFUNDS': {'name': 'Federal Funds Rate', 'unit': '%', 'base': 3.0},
        'CPIAUCSL': {'name': 'Consumer Price Index', 'unit': 'Index', 'base': 280},
        'GDP': {'name': 'GDP', 'unit': 'Billions', 'base': 25000}
    }
    
    for series_id, info in indicators.items():
        if series_id == 'UNRATE':
            values = info['base'] + np.random.normal(0, 0.3, len(dates))
            values[10:20] = 8.0 + np.random.normal(0, 1.0, 10)
        elif series_id == 'FEDFUNDS':
            values = np.concatenate([
                np.full(24, 0.5),
                np.linspace(0.5, 5.0, len(dates)-24)
            ])
        else:
            values = info['base'] * (1.02 ** (np.arange(len(dates)) / 12)) + np.random.normal(0, info['base']*0.01, len(dates))
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'value': values[i],
                'series_id': series_id,
                'series_name': info['name'],
                'unit': info['unit']
            })
    
    return pd.DataFrame(data)

def create_dashboard_overview(df):
    indicators = ['UNRATE', 'CPIAUCSL', 'GDP', 'FEDFUNDS']
    titles = ['Unemployment Rate (%)', 'Consumer Price Index', 'GDP (Billions)', 'Federal Funds Rate (%)']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, indicator in enumerate(indicators):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        series_data = df[df['series_id'] == indicator].sort_values('date')
        
        fig.add_trace(
            go.Scatter(
                x=series_data['date'],
                y=series_data['value'],
                mode='lines',
                name=series_data['series_name'].iloc[0],
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=row, col=col
        )
        
        latest = series_data.iloc[-1]
        fig.add_annotation(
            x=latest['date'],
            y=latest['value'],
            text=f"{latest['value']:.1f}",
            showarrow=False,
            bgcolor="white",
            bordercolor=colors[i],
            borderwidth=1,
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text="Economic Dashboard Overview",
        title_x=0.5,
        template='plotly_white'
    )
    
    return fig

st.markdown('<div class="main-header">📊 Economic Pulse Dashboard</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    
    page = st.selectbox(
        "Choose a view:",
        ["📊 Overview", "📈 Individual Charts", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Data Info")
    st.markdown("Sample Economic Data")
    st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

df = create_comprehensive_data()

if page == "📊 Overview":
    st.markdown("### 📈 Key Economic Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'GDP']
    for i, metric_id in enumerate(metrics):
        metric_data = df[df['series_id'] == metric_id].sort_values('date')
        if not metric_data.empty:
            latest_value = metric_data.iloc[-1]['value']
            prev_value = metric_data.iloc[-2]['value'] if len(metric_data) > 1 else latest_value
            change = ((latest_value - prev_value) / prev_value * 100) if prev_value != 0 else 0
            
            with [col1, col2, col3, col4][i]:
                st.metric(
                    label=metric_data.iloc[-1]['series_name'],
                    value=f"{latest_value:.1f}{metric_data.iloc[-1]['unit']}",
                    delta=f"{change:+.1f}%"
                )
    
    st.markdown("---")
    overview_chart = create_dashboard_overview(df)
    st.plotly_chart(overview_chart, use_container_width=True)
    
elif page == "📈 Individual Charts":
    st.markdown("### 📈 Individual Economic Indicators")
    
    available_indicators = df['series_id'].unique()
    selected_indicator = st.selectbox(
        "Select an indicator:",
        available_indicators,
        format_func=lambda x: df[df['series_id'] == x]['series_name'].iloc[0]
    )
    
    series_data = df[df['series_id'] == selected_indicator].sort_values('date')
    
    fig = px.line(
        series_data,
        x='date',
        y='value',
        title=f"{series_data['series_name'].iloc[0]} Over Time",
        labels={'value': f"Value ({series_data['unit'].iloc[0]})", 'date': 'Date'}
    )
    
    fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    latest_value = series_data.iloc[-1]['value']
    prev_value = series_data.iloc[-2]['value'] if len(series_data) > 1 else latest_value
    change = ((latest_value - prev_value) / prev_value * 100) if prev_value != 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Value", f"{latest_value:.2f} {series_data.iloc[-1]['unit']}")
    with col2:
        st.metric("Period Change", f"{change:+.1f}%")
    with col3:
        st.metric("Data Points", len(series_data))

elif page == "ℹ️ About":
    st.markdown("""
    ### ℹ️ About Economic Pulse Dashboard
    
    **Economic Pulse** provides real-time economic monitoring with:
    
    - 📊 **Multi-indicator Dashboard**: Track unemployment, inflation, GDP, and interest rates
    - 📈 **Interactive Visualizations**: Dynamic charts with Plotly
    - 🎛️ **Multiple Views**: Overview and detailed individual analysis
    - 📱 **Responsive Design**: Works on all devices
    
    #### Sample Data Includes:
    - Unemployment Rate with COVID impact
    - Federal Reserve interest rate cycles  
    - GDP growth trends
    - Consumer Price Index inflation tracking
    
    *Built with Streamlit, Pandas, and Plotly for professional economic analysis*
    """)

st.success("✅ Economic Pulse Dashboard - Fully Operational!")
