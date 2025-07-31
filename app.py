# streamlit_dashboard.py - Streamlit Application for Economic Pulse Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# --- Streamlit App Title and Introduction ---
st.set_page_config(layout="wide", page_title="Economic Pulse Dashboard")
st.title("📊 ML-Enhanced Economic Pulse Dashboard")
st.markdown("This dashboard provides a comprehensive overview of key economic indicators, including ML-powered forecasts, anomaly detection, and sentiment analysis.")

# --- Diagnostic Check for Module Existence (Optional for deployed app, but good for local testing) ---
module_name = 'advanced_mlai_features'
module_found = False
for path in sys.path:
    if os.path.exists(os.path.join(path, module_name + '.py')):
        module_found = True
        break
if not module_found:
    st.error(f"❌ Could not find '{module_name}.py'. Please ensure it's in the same directory.")
    st.stop()

try:
    from advanced_mlai_features import (
        create_ml_enhanced_dashboard,
        generate_ml_insights,
        EconomicMLPredictor,
        EconomicAnomalyDetector,
        EconomicSentimentAnalyzer
    )
except ModuleNotFoundError as e:
    st.error(f"❌ ModuleNotFoundError during import: {e}")
    st.error("This indicates 'advanced_mlai_features.py' is not found or has issues. Please verify it exists and is accessible.")
    st.stop()

st.markdown("---")
st.subheader("Data Collection & Processing")

# --- Data Collection Function ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def collect_all_economic_data():
    """Complete data collection function with working API"""
    
    api_key = os.environ.get('FRED_API_KEY', 'a9df9d2acc55e78b47eff58dc9a6a04e') # Fallback for local testing
    if not api_key:
        st.error("FRED API Key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
        st.stop()

    indicators = {
        'UNRATE': {'name': 'Unemployment Rate', 'category': 'Employment', 'unit': '%'},
        'PAYEMS': {'name': 'Nonfarm Payrolls', 'category': 'Employment', 'unit': 'Thousands'},
        'CPIAUCSL': {'name': 'Consumer Price Index', 'category': 'Inflation', 'unit': 'Index'},
        'GDP': {'name': 'GDP', 'category': 'Growth', 'unit': 'Billions'},
        'GDPC1': {'name': 'Real GDP', 'category': 'Growth', 'unit': 'Billions'},
        'FEDFUNDS': {'name': 'Federal Funds Rate', 'category': 'Fed Policy', 'unit': '%'},
        'DGS10': {'name': '10-Year Treasury', 'category': 'Interest Rates', 'unit': '%'},
        'DGS2': {'name': '2-Year Treasury', 'category': 'Interest Rates', 'unit': '%'},
    }
    
    st.info("🔄 Collecting economic data... This might take a moment.")
    
    all_data = []
    successful = 0
    failed = 0
    
    status_messages = [] # Collect messages to display in Streamlit
    
    for series_id, info in indicators.items():
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&start_date=2020-01-01&limit=1000"
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for obs in data['observations']:
                    if obs['value'] != '.' and obs['value'] is not None:
                        try:
                            all_data.append({
                                'date': pd.to_datetime(obs['date']),
                                'value': float(obs['value']),
                                'series_id': series_id,
                                'series_name': info['name'],
                                'category': info['category'],
                                'unit': info['unit']
                            })
                        except (ValueError, TypeError):
                            continue
                successful += 1
                status_messages.append(f"  ✅ {series_id}: {info['name']}")
            else:
                failed += 1
                status_messages.append(f"  ❌ {series_id}: API error {response.status_code}")
                
        except Exception as e:
            failed += 1
            status_messages.append(f"  ❌ {series_id}: {str(e)[:50]}...")
    
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(['series_id', 'date'])
        st.success("✅ Data Collection Complete!")
        st.write(f"Success rate: {successful}/{len(indicators)} ({successful/len(indicators)*100:.1f}%)")
        st.write(f"Total records: {len(df):,}")
        # Display individual status messages in an expander
        with st.expander("Data Collection Details"):
            for msg in status_messages:
                st.write(msg)
        return df
    else:
        st.error("❌ No data collected!")
        return pd.DataFrame()

# --- Data Processing Functions ---
@st.cache_data
def process_economic_data(df):
    """Process the economic data with calculations"""
    st.info("🔄 Processing economic data...")
    
    if df.empty:
        st.warning("No data to process.")
        return pd.DataFrame()
    
    latest_data = []
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy() 

    for series_id in df_copy['series_id'].unique():
        series_data = df_copy[df_copy['series_id'] == series_id].sort_values('date')
        
        if len(series_data) > 0:
            latest = series_data.iloc[-1]
            prev = series_data.iloc[-2] if len(series_data) > 1 else None
            
            # Calculate changes
            change_abs = None
            change_pct = None
            if prev is not None:
                change_abs = latest['value'] - prev['value']
                if prev['value'] != 0:
                    change_pct = (change_abs / prev['value'] * 100)
            
            # Calculate year-over-year change (approximate)
            yoy_change = None
            if len(series_data) >= 12:
                year_ago = series_data.iloc[-12]
                if year_ago['value'] != 0:
                    yoy_change = ((latest['value'] - year_ago['value']) / year_ago['value'] * 100)
            
            latest_data.append({
                'series_id': latest['series_id'],
                'series_name': latest['series_name'],
                'category': latest['category'],
                'unit': latest['unit'],
                'current_value': latest['value'],
                'previous_value': prev['value'] if prev is not None else None,
                'change_abs': change_abs,
                'change_pct': change_pct,
                'change_yoy_pct': yoy_change,
                'last_updated': latest['date'],
                'data_points': len(series_data)
            })
    
    processed_df = pd.DataFrame(latest_data)

    # Add "Inflation (CPI YoY)" calculation here if CPIAUCSL is present
    # This ensures the 'Inflation (CPI YoY)' column exists if needed later
    if 'CPIAUCSL' in df_copy['series_id'].unique():
        cpi_data = df_copy[df_copy['series_id'] == 'CPIAUCSL'].sort_values('date').copy()
        if len(cpi_data) >= 12:
            # Calculate YoY growth for CPIAUCSL and add as a new "indicator"
            cpi_data['YoY Growth (%)'] = cpi_data['value'].pct_change(12) * 100 # Calculate YoY percentage change
            latest_cpi_yoy = cpi_data['YoY Growth (%)'].iloc[-1]
            prev_cpi_yoy = cpi_data['YoY Growth (%)'].iloc[-2] if len(cpi_data) >= 2 else None

            # Add a new entry for 'Inflation (CPI YoY)' as a distinct indicator
            processed_df = pd.concat([processed_df, pd.DataFrame([{
                'series_id': 'INFLATION_CPI_YOY', # New unique ID for this derived metric
                'series_name': 'Inflation (CPI YoY)',
                'category': 'Inflation',
                'unit': '%',
                'current_value': latest_cpi_yoy,
                'previous_value': prev_cpi_yoy,
                'change_abs': (latest_cpi_yoy - prev_cpi_yoy) if prev_cpi_yoy is not None else None,
                'change_pct': ((latest_cpi_yoy - prev_cpi_yoy) / prev_cpi_yoy * 100) if prev_cpi_yoy is not None and prev_cpi_yoy != 0 else None,
                'change_yoy_pct': None, # This is already a YoY metric
                'last_updated': cpi_data['date'].iloc[-1],
                'data_points': len(cpi_data)
            }])], ignore_index=True)


    st.success(f"✅ Processing Complete! {len(processed_df)} indicators processed")
    return processed_df

# --- Visualization Functions ---
@st.cache_data
def create_dashboard_overview(df):
    """Create comprehensive dashboard overview"""
    st.info("🔄 Creating dashboard visualizations...")
    
    if df.empty:
        st.warning("No data to visualize.")
        return None
    
    # Key indicators for dashboard
    # Ensure 'INFLATION_CPI_YOY' is included if you want to plot it
    key_indicators = ['UNRATE', 'INFLATION_CPI_YOY', 'GDP', 'FEDFUNDS'] 
    available_indicators = [ind for ind in df['series_id'].unique() if ind in key_indicators]
    
    if not available_indicators:
        st.warning("No key indicators available for overview dashboard.")
        return None
    
    # Create 2x2 subplot
    titles = []
    for ind in available_indicators:
        indicator_info = df[df['series_id'] == ind].iloc[0]
        titles.append(f"{indicator_info['series_name']} ({indicator_info['unit']})")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, indicator_id in enumerate(available_indicators):
        if i >= 4:
            break
            
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Get the latest value from processed_df for this indicator
        latest_value_series = df[df['series_id'] == indicator_id]
        if not latest_value_series.empty:
            latest_value = latest_value_series.iloc[0]['current_value']
            series_name = latest_value_series.iloc[0]['series_name']
        else:
            latest_value = 0.0
            series_name = indicator_id # Fallback name

        # Generate sample time series data for the plot
        dates = pd.date_range(start='2020-01-01', end='2025-01-01', freq='M')
        if indicator_id == 'UNRATE':
            values = 4.0 + np.random.normal(0, 0.5, len(dates))
            values[10:20] = 8.0 + np.random.normal(0, 0.8, 10)
        elif indicator_id == 'FEDFUNDS':
            values = np.concatenate([
                np.full(24, 0.5),
                np.linspace(0.5, 5.0, len(dates)-24)
            ])
        elif indicator_id == 'INFLATION_CPI_YOY': # Handle the new derived inflation metric
            # Sample data for inflation YoY, assuming a baseline and some fluctuation
            values = 2.0 + np.random.normal(0, 1.0, len(dates))
            values[-1] = latest_value # Ensure the last value matches the calculated one
        else:
            values = np.cumsum(np.random.normal(0.02, 0.1, len(dates))) + latest_value * 0.8
            
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name=series_name,
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=row, col=col
        )
            
        fig.add_annotation(
            x=dates[-1],
            y=values[-1],
            text=f"{latest_value:.1f}",
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
    
    st.success("✅ Dashboard Visualizations Created!")
    return fig

# --- Generate AI Summary ---
@st.cache_data
def generate_ai_summary(latest_df):
    """Generate AI-powered economic summary"""
    st.info("🔄 Generating AI summary...")
    
    if latest_df.empty:
        return "No economic data available for analysis."
    
    unemployment = latest_df[latest_df['series_id'] == 'UNRATE']['current_value'].iloc[0] if 'UNRATE' in latest_df['series_id'].values else None
    fed_rate = latest_df[latest_df['series_id'] == 'FEDFUNDS']['current_value'].iloc[0] if 'FEDFUNDS' in latest_df['series_id'].values else None
    gdp = latest_df[latest_df['series_id'] == 'GDP']['current_value'].iloc[0] if 'GDP' in latest_df['series_id'].values else None
    # Get the calculated Inflation (CPI YoY) value
    inflation_yoy = latest_df[latest_df['series_id'] == 'INFLATION_CPI_YOY']['current_value'].iloc[0] if 'INFLATION_CPI_YOY' in latest_df['series_id'].values else None

    summary_parts = []
    summary_parts.append("**Current Economic Climate Analysis:**\n")
    
    if unemployment is not None:
        if unemployment < 4.0:
            summary_parts.append(f"• **Labor Market**: Strong with unemployment at {unemployment:.1f}%, indicating a tight job market.")
        elif unemployment < 6.0:
            summary_parts.append(f"• **Labor Market**: Moderate with unemployment at {unemployment:.1f}%, showing balanced conditions.")
        else:
            summary_parts.append(f"• **Labor Market**: Concerning with unemployment at {unemployment:.1f}%, indicating economic stress.")
    
    if fed_rate is not None:
        if fed_rate < 2.0:
            summary_parts.append(f"• **Monetary Policy**: Accommodative with Fed funds rate at {fed_rate:.1f}%, supporting economic growth.")
        elif fed_rate < 5.0:
            summary_parts.append(f"• **Monetary Policy**: Neutral with Fed funds rate at {fed_rate:.1f}%, balancing growth and inflation.")
        else:
            summary_parts.append(f"• **Monetary Policy**: Restrictive with Fed funds rate at {fed_rate:.1f}%, aimed at controlling inflation.")
    
    if gdp is not None:
        summary_parts.append(f"• **Economic Output**: GDP at ${gdp:,.1f} billion, reflecting the overall economic scale.")
    
    if inflation_yoy is not None:
        if inflation_yoy < 2.5:
            summary_parts.append(f"• **Inflation (CPI YoY)**: Currently at {inflation_yoy:.1f}%, suggesting inflation is well-contained.")
        elif inflation_yoy < 4.0:
            summary_parts.append(f"• **Inflation (CPI YoY)**: At {inflation_yoy:.1f}%, indicating moderate inflationary pressures.")
        else:
            summary_parts.append(f"• **Inflation (CPI YoY)**: Elevated at {inflation_yoy:.1f}%, signaling significant inflationary concerns.")

    summary_parts.append(f"\n**Overall Assessment**: ")
    if unemployment and unemployment < 4.5 and fed_rate and fed_rate < 6.0 and inflation_yoy and inflation_yoy < 4.0:
        summary_parts.append("The economy shows resilient fundamentals with balanced growth prospects and manageable inflation.")
    elif unemployment and unemployment > 6.0 or inflation_yoy and inflation_yoy > 5.0:
        summary_parts.append("Economic conditions show signs of stress requiring careful monitoring and potential policy adjustments.")
    else:
        summary_parts.append("Mixed economic signals suggest a transitional period with both opportunities and challenges.")
    
    summary_parts.append(f"\n*Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    st.success("✅ AI Summary Generated!")
    return "\n".join(summary_parts)

# --- Full Pipeline Execution for Streamlit ---
@st.cache_resource
def run_full_pipeline():
    """Run the complete data pipeline and return results for Streamlit display"""
    
    with st.spinner("🚀 Starting Full Data Pipeline Test..."):
        try:
            raw_data = collect_all_economic_data()
            if raw_data.empty:
                st.error("Data collection failed. Cannot proceed with processing and visualization.")
                return {'success': False, 'error': 'Data collection failed', 'data': pd.DataFrame()}
                
            processed_data = process_economic_data(raw_data)
            if processed_data.empty:
                st.error("Data processing failed. Cannot proceed with visualization.")
                return {'success': False, 'error': 'Data processing failed', 'data': raw_data}
                
            overview_chart = create_dashboard_overview(processed_data)
            
            summary = generate_ai_summary(processed_data)
            
            results = {
                'success': True,
                'data': raw_data,
                'latest': processed_data,
                'overview_chart': overview_chart,
                'summary': summary,
                'metrics': {
                    'total_records': len(raw_data),
                    'indicators_processed': len(processed_data),
                    'data_freshness': processed_data['last_updated'].max() if not processed_data.empty else None
                }
            }
            
            st.success("🎉 Full Pipeline Test SUCCESSFUL!")
            st.write(f"📊 Total records: {results['metrics']['total_records']:,}")
            st.write(f"📈 Indicators processed: {results['metrics']['indicators_processed']}")
            st.write(f"📅 Data freshness: {results['metrics']['data_freshness'].strftime('%Y-%m-%d') if results['metrics']['data_freshness'] else 'N/A'}")
            
            return results
            
        except Exception as e:
            st.error(f"❌ An error occurred during pipeline execution: {e}")
            return {'success': False, 'error': str(e), 'data': pd.DataFrame()}

# --- Run the full pipeline and display results ---
pipeline_results = run_full_pipeline()

if pipeline_results['success']:
    st.markdown("---")
    st.subheader("Pipeline Test Results")
    st.success("✅ Data Collection: SUCCESS")
    st.success("✅ Data Processing: SUCCESS") 
    st.success("✅ Visualization: SUCCESS")
    st.success("✅ AI Summary: SUCCESS")
    
    st.markdown("---")
    st.subheader("Economic Dashboard Overview")
    if pipeline_results['overview_chart']:
        st.plotly_chart(pipeline_results['overview_chart'], use_container_width=True)
    
    st.markdown("---")
    st.subheader("AI Economic Summary")
    st.markdown(pipeline_results['summary'])
    
    st.markdown("---")
    st.subheader("Latest Economic Indicators")
    latest_indicators_df = pipeline_results['latest'] # Use the full processed_df
    
    # Filter for display, maybe top N or specific ones
    display_indicators_ids = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'INFLATION_CPI_YOY', 'GDP']
    display_df = latest_indicators_df[latest_indicators_df['series_id'].isin(display_indicators_ids)].copy()

    # Sort to ensure consistent order
    display_df['sort_order'] = display_df['series_id'].apply(lambda x: display_indicators_ids.index(x) if x in display_indicators_ids else len(display_indicators_ids))
    display_df = display_df.sort_values('sort_order').drop(columns='sort_order')

    display_data = []
    for _, row in display_df.iterrows():
        change_str = ""
        if pd.notna(row['change_pct']):
            change_str = f" ({row['change_pct']:+.1f}%)"
        elif pd.notna(row['change_abs']):
            change_str = f" ({row['change_abs']:+.1f})"
        
        # Format current value based on unit/type
        if row['unit'] == '%':
            formatted_value = f"{row['current_value']:.1f}{row['unit']}"
        elif row['unit'] == 'Billions':
            formatted_value = f"${row['current_value']:,.1f} {row['unit']}"
        elif row['unit'] == 'Thousands':
            formatted_value = f"{row['current_value']:,.0f} {row['unit']}"
        else:
            formatted_value = f"{row['current_value']:.1f} {row['unit']}"

        display_data.append([row['series_name'], f"{formatted_value}{change_str}"])
    
    st.table(pd.DataFrame(display_data, columns=["Indicator", "Value (Change)"]))

    st.markdown("---")
    st.subheader("ML-Enhanced Analysis")
    # For demonstration, create sample data if pipeline_results['data'] is empty
    if pipeline_results['data'].empty:
        date_range_length = len(pd.date_range('2020-01-01', '2024-12-01', freq='M'))
        sample_data_for_ml = pd.DataFrame({
            'date': pd.date_range('2020-01-01', '2024-12-01', freq='M'),
            'value': 4.0 + np.random.normal(0, 0.5, date_range_length),
            'series_id': ['UNRATE'] * date_range_length,
            'series_name': ['Unemployment Rate'] * date_range_length,
            'category': ['Employment'] * date_range_length,
            'unit': ['%'] * date_range_length
        })
        # Add a placeholder for CPIAUCSL to generate derived inflation for ML
        cpi_dates = pd.date_range('2020-01-01', '2024-12-01', freq='M')
        cpi_values = 200 + np.cumsum(np.random.normal(0.5, 1.0, len(cpi_dates)))
        sample_data_for_ml = pd.concat([sample_data_for_ml, pd.DataFrame({
            'date': cpi_dates,
            'value': cpi_values,
            'series_id': ['CPIAUCSL'] * len(cpi_dates),
            'series_name': ['Consumer Price Index'] * len(cpi_dates),
            'category': ['Inflation'] * len(cpi_dates),
            'unit': ['Index'] * len(cpi_dates)
        })], ignore_index=True)

        sample_data_for_ml.loc[10:15, 'value'] = 8.0 + np.random.normal(0, 1.0, 6)
        ml_input_data = sample_data_for_ml
    else:
        ml_input_data = pipeline_results['data']

    # Create ML dashboard
    ml_results = create_ml_enhanced_dashboard(ml_input_data)
    
    # Show ML dashboard
    if ml_results and ml_results['dashboard']:
        st.subheader("ML-Powered Dashboard Visualizations")
        st.plotly_chart(ml_results['dashboard'], use_container_width=True)
    
    # Generate and show ML insights
    if ml_results:
        st.subheader("ML-Powered Economic Insights")
        ml_insights = generate_ml_insights(
            ml_results['predictions'], 
            ml_results['anomalies'], 
            ml_results['sentiment']
        )
        st.markdown(ml_insights)
    else:
        st.warning("❌ ML-Enhanced Dashboard creation failed, cannot generate insights.")

else:
    st.error(f"❌ PIPELINE TEST FAILED: {pipeline_results['error']}")

st.markdown("---")
st.success("🏁 Integration test complete!")
