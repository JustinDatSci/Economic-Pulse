# 🎨 Enhanced UI Components for Modern Dashboard Experience
# Advanced Streamlit UI components with modern design patterns

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import base64
import io

class ModernUIComponents:
    """Modern UI components for enhanced user experience"""
    
    def __init__(self):
        self.load_custom_css()
        
    def load_custom_css(self):
        """Load custom CSS for modern styling"""
        
        st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Modern Header Styles */
        .modern-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .modern-header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .modern-header p {
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        /* Card Styles */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 48px rgba(0,0,0,0.15);
        }
        
        .asset-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .asset-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 32px rgba(0,0,0,0.1);
            border-color: #667eea;
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%);
            border: 1px solid #e9d5ff;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(139, 92, 246, 0.1);
        }
        
        .alert-card-high {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left: 6px solid #dc2626;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(220, 38, 38, 0.1);
        }
        
        .alert-card-medium {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-left: 6px solid #d97706;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(217, 119, 6, 0.1);
        }
        
        .alert-card-low {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 6px solid #16a34a;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 16px rgba(22, 163, 74, 0.1);
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        
        /* Metric Styling */
        .metric-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }
        
        .metric-container:hover {
            transform: scale(1.02);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 1rem;
            color: #6b7280;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-delta {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        .metric-delta.positive {
            color: #10b981;
        }
        
        .metric-delta.negative {
            color: #ef4444;
        }
        
        /* Progress Bar Styling */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
        }
        
        /* Chart Container */
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
        
        /* Loading Animation */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 3rem;
        }
        
        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e2e8f0;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Status Badge */
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-online {
            background: #d1fae5;
            color: #047857;
        }
        
        .status-offline {
            background: #fee2e2;
            color: #dc2626;
        }
        
        .status-warning {
            background: #fef3c7;
            color: #d97706;
        }
        
        /* Notification Toast */
        .notification-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
        }
        
        /* Data Table Styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        }
        
        /* Info Box */
        .info-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .info-box-success {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-color: #bbf7d0;
        }
        
        .info-box-warning {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border-color: #fde68a;
        }
        
        .info-box-error {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-color: #fecaca;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_modern_header(self, title, subtitle=""):
        """Create modern header component"""
        
        st.markdown(f"""
        <div class="modern-header">
            <h1>{title}</h1>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def create_metric_card(self, title, value, delta=None, delta_color=None):
        """Create modern metric card"""
        
        delta_class = ""
        if delta and delta_color:
            delta_class = f"metric-delta {delta_color}"
        elif delta:
            delta_class = "metric-delta positive" if "+" in str(delta) else "metric-delta negative"
        
        delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ''
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{title}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_status_badge(self, status, text):
        """Create status badge"""
        
        status_class = f"status-badge status-{status}"
        
        st.markdown(f"""
        <span class="{status_class}">{text}</span>
        """, unsafe_allow_html=True)
    
    def create_alert_card(self, severity, title, message, timestamp=None):
        """Create alert card with severity styling"""
        
        card_class = f"alert-card-{severity}"
        timestamp_html = f"<small>{timestamp}</small>" if timestamp else ""
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">{title}</h4>
            <p style="margin: 0.5rem 0; color: #4b5563;">{message}</p>
            {timestamp_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_info_box(self, message, box_type="info"):
        """Create styled info box"""
        
        box_class = f"info-box info-box-{box_type}"
        
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        icon = icons.get(box_type, "ℹ️")
        
        st.markdown(f"""
        <div class="{box_class}">
            {icon} {message}
        </div>
        """, unsafe_allow_html=True)
    
    def create_loading_animation(self, text="Loading..."):
        """Create loading animation"""
        
        st.markdown(f"""
        <div class="loading-container">
            <div style="text-align: center;">
                <div class="loading-spinner"></div>
                <p style="margin-top: 1rem; color: #6b7280;">{text}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_progress_card(self, title, current, total, description=""):
        """Create progress card with percentage"""
        
        percentage = (current / total * 100) if total > 0 else 0
        
        st.markdown(f"""
        <div class="asset-card">
            <h4 style="margin: 0 0 1rem 0;">{title}</h4>
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>{current} / {total}</span>
                    <span>{percentage:.1f}%</span>
                </div>
            </div>
            {f'<p style="color: #6b7280; margin: 0;">{description}</p>' if description else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Add progress bar
        st.progress(percentage / 100)

class EnhancedChartComponents:
    """Enhanced chart components with modern styling"""
    
    def __init__(self):
        self.default_colors = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c',
            '#4facfe', '#00f2fe', '#43e97b', '#38f9d7',
            '#ffecd2', '#fcb69f', '#a8edea', '#fed6e3'
        ]
    
    def create_modern_line_chart(self, df, x, y, title="", color=None, height=400):
        """Create modern line chart with enhanced styling"""
        
        fig = px.line(df, x=x, y=y, color=color, title=title)
        
        fig.update_layout(
            template='plotly_white',
            height=height,
            title_font_size=20,
            title_font_family='Inter',
            title_x=0.5,
            font_family='Inter',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            )
        )
        
        # Update line styling
        for i, trace in enumerate(fig.data):
            trace.line.width = 3
            trace.line.color = self.default_colors[i % len(self.default_colors)]
        
        return fig
    
    def create_modern_bar_chart(self, df, x, y, title="", color=None, height=400):
        """Create modern bar chart with enhanced styling"""
        
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        
        fig.update_layout(
            template='plotly_white',
            height=height,
            title_font_size=20,
            title_font_family='Inter',
            title_x=0.5,
            font_family='Inter',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            )
        )
        
        # Update bar styling
        if not color:
            fig.update_traces(
                marker_color=self.default_colors[0],
                marker_line_color='rgba(0,0,0,0)',
                marker_line_width=0
            )
        
        return fig
    
    def create_modern_scatter_plot(self, df, x, y, title="", color=None, size=None, height=400):
        """Create modern scatter plot with enhanced styling"""
        
        fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title)
        
        fig.update_layout(
            template='plotly_white',
            height=height,
            title_font_size=20,
            title_font_family='Inter',
            title_x=0.5,
            font_family='Inter',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            )
        )
        
        # Update marker styling
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=2,
            marker_size=10
        )
        
        return fig
    
    def create_modern_pie_chart(self, labels, values, title="", height=400):
        """Create modern pie chart with enhanced styling"""
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            marker_colors=self.default_colors
        )])
        
        fig.update_layout(
            template='plotly_white',
            height=height,
            title=title,
            title_font_size=20,
            title_font_family='Inter',
            title_x=0.5,
            font_family='Inter',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        return fig
    
    def create_modern_heatmap(self, df, title="", height=400):
        """Create modern heatmap with enhanced styling"""
        
        fig = px.imshow(
            df,
            title=title,
            color_continuous_scale='RdBu',
            aspect='auto',
            text_auto=True
        )
        
        fig.update_layout(
            template='plotly_white',
            height=height,
            title_font_size=20,
            title_font_family='Inter',
            title_x=0.5,
            font_family='Inter'
        )
        
        return fig
    
    def create_candlestick_chart(self, df, title="", height=400):
        """Create modern candlestick chart"""
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        )])
        
        fig.update_layout(
            template='plotly_white',
            height=height,
            title=title,
            title_font_size=20,
            title_font_family='Inter',
            title_x=0.5,
            font_family='Inter',
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            )
        )
        
        return fig

class InteractiveDashboard:
    """Interactive dashboard with modern components"""
    
    def __init__(self):
        self.ui = ModernUIComponents()
        self.charts = EnhancedChartComponents()
    
    def create_sidebar_navigation(self):
        """Create modern sidebar navigation"""
        
        with st.sidebar:
            st.markdown("### 🎛️ Navigation")
            
            # Main navigation
            pages = {
                "🏠 Dashboard": "dashboard",
                "📊 Analytics": "analytics", 
                "💼 Portfolio": "portfolio",
                "🔮 Predictions": "predictions",
                "🚨 Alerts": "alerts",
                "⚙️ Settings": "settings"
            }
            
            selected_page = st.selectbox(
                "Select Page:",
                list(pages.keys()),
                key="main_nav"
            )
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            if st.button("📧 Send Report", use_container_width=True):
                self.ui.create_info_box("Report sent successfully!", "success")
            
            if st.button("💾 Export Data", use_container_width=True):
                self.ui.create_info_box("Data export initiated...", "info")
            
            st.markdown("---")
            
            # System status
            st.markdown("### 📡 System Status")
            
            col1, col2 = st.columns(2)
            with col1:
                self.ui.create_status_badge("online", "API")
            with col2:
                self.ui.create_status_badge("online", "DB")
            
            return pages[selected_page]
    
    def create_dashboard_header(self, title, subtitle=""):
        """Create dashboard header with modern styling"""
        
        self.ui.create_modern_header(title, subtitle)
        
        # Add last updated info
        st.markdown(f"""
        <div style="text-align: center; color: #6b7280; margin-bottom: 2rem;">
            📅 Last Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        </div>
        """, unsafe_allow_html=True)
    
    def create_kpi_section(self, metrics_data):
        """Create KPI section with modern metric cards"""
        
        st.markdown("### 📊 Key Performance Indicators")
        
        cols = st.columns(len(metrics_data))
        
        for i, (title, value, delta) in enumerate(metrics_data):
            with cols[i]:
                self.ui.create_metric_card(title, value, delta)
    
    def create_chart_section(self, chart_data, title="Charts"):
        """Create chart section with modern styling"""
        
        st.markdown(f"### {title}")
        
        # Chart tabs
        chart_types = list(chart_data.keys())
        tabs = st.tabs([f"📈 {chart}" for chart in chart_types])
        
        for i, (chart_name, data) in enumerate(chart_data.items()):
            with tabs[i]:
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    if data['type'] == 'line':
                        fig = self.charts.create_modern_line_chart(
                            data['df'], data['x'], data['y'], data.get('title', chart_name)
                        )
                    elif data['type'] == 'bar':
                        fig = self.charts.create_modern_bar_chart(
                            data['df'], data['x'], data['y'], data.get('title', chart_name)
                        )
                    elif data['type'] == 'scatter':
                        fig = self.charts.create_modern_scatter_plot(
                            data['df'], data['x'], data['y'], data.get('title', chart_name)
                        )
                    elif data['type'] == 'pie':
                        fig = self.charts.create_modern_pie_chart(
                            data['labels'], data['values'], data.get('title', chart_name)
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    def create_data_table_section(self, df, title="Data"):
        """Create data table section with modern styling"""
        
        st.markdown(f"### 📋 {title}")
        
        # Table controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search", placeholder="Search data...")
        
        with col2:
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
        
        with col3:
            if st.button("📊 Export CSV"):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Filter data if search term provided
        if search_term:
            df_filtered = df[df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)]
        else:
            df_filtered = df
        
        # Display table with pagination
        total_rows = len(df_filtered)
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        if total_pages > 1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
            start_idx = page * rows_per_page
            end_idx = start_idx + rows_per_page
            df_page = df_filtered.iloc[start_idx:end_idx]
        else:
            df_page = df_filtered
        
        st.dataframe(df_page, use_container_width=True)
        
        # Pagination info
        if total_pages > 1:
            st.markdown(f"Showing {len(df_page)} of {total_rows} rows (Page {page + 1} of {total_pages})")

class NotificationManager:
    """Modern notification system"""
    
    def __init__(self):
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
    
    def add_notification(self, message, notification_type="info", duration=5):
        """Add notification to queue"""
        
        notification = {
            'id': len(st.session_state.notifications),
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now(),
            'duration': duration
        }
        
        st.session_state.notifications.append(notification)
    
    def display_notifications(self):
        """Display active notifications"""
        
        current_time = datetime.now()
        active_notifications = []
        
        for notification in st.session_state.notifications:
            time_diff = (current_time - notification['timestamp']).seconds
            
            if time_diff < notification['duration']:
                active_notifications.append(notification)
        
        st.session_state.notifications = active_notifications
        
        # Display notifications
        for notification in active_notifications:
            self._display_toast(notification)
    
    def _display_toast(self, notification):
        """Display individual toast notification"""
        
        icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        icon = icons.get(notification['type'], 'ℹ️')
        
        st.markdown(f"""
        <div class="notification-toast">
            {icon} {notification['message']}
        </div>
        """, unsafe_allow_html=True)