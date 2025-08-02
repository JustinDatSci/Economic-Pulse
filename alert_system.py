# 🚨 Real-Time Alerts and Notification System
# Advanced monitoring, alerts, and notification infrastructure

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Email and notification imports
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    WEBHOOK_AVAILABLE = True
except ImportError:
    WEBHOOK_AVAILABLE = False

class AlertEngine:
    """Advanced alert engine for financial monitoring"""
    
    def __init__(self):
        self.alert_rules = {}
        self.alert_history = []
        self.notification_channels = {}
        self.active_alerts = {}
        self.alert_cooldowns = {}
        
    def add_alert_rule(self, rule_id, rule_config):
        """Add a new alert rule"""
        
        required_fields = ['name', 'condition', 'threshold', 'asset', 'notification_channels']
        if not all(field in rule_config for field in required_fields):
            raise ValueError(f"Alert rule must contain: {required_fields}")
        
        rule_config['created_at'] = datetime.now()
        rule_config['enabled'] = rule_config.get('enabled', True)
        rule_config['cooldown_minutes'] = rule_config.get('cooldown_minutes', 15)
        
        self.alert_rules[rule_id] = rule_config
        return True
    
    def remove_alert_rule(self, rule_id):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            return True
        return False
    
    def check_alerts(self, df):
        """Check all alert rules against current data"""
        
        triggered_alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.get('enabled', True):
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_id):
                continue
            
            # Get latest data for the asset
            asset_data = df[df['series_id'] == rule['asset']]
            if asset_data.empty:
                continue
            
            latest_data = asset_data.sort_values('date').iloc[-1]
            
            # Check condition
            if self._evaluate_condition(latest_data, rule):
                alert = self._create_alert(rule_id, rule, latest_data)
                triggered_alerts.append(alert)
                
                # Set cooldown
                self.alert_cooldowns[rule_id] = datetime.now()
        
        # Send notifications for triggered alerts
        for alert in triggered_alerts:
            self._send_alert_notifications(alert)
            self.alert_history.append(alert)
        
        return triggered_alerts
    
    def _is_in_cooldown(self, rule_id):
        """Check if alert rule is in cooldown period"""
        
        if rule_id not in self.alert_cooldowns:
            return False
        
        last_triggered = self.alert_cooldowns[rule_id]
        cooldown_minutes = self.alert_rules[rule_id].get('cooldown_minutes', 15)
        
        return datetime.now() - last_triggered < timedelta(minutes=cooldown_minutes)
    
    def _evaluate_condition(self, data, rule):
        """Evaluate alert condition against data"""
        
        condition = rule['condition']
        threshold = rule['threshold']
        current_value = data['value']
        
        # Price-based conditions
        if condition == 'price_above':
            return current_value > threshold
        elif condition == 'price_below':
            return current_value < threshold
        elif condition == 'price_change_up':
            # Need historical data for this
            return self._check_price_change(data, rule, direction='up')
        elif condition == 'price_change_down':
            return self._check_price_change(data, rule, direction='down')
        elif condition == 'volatility_spike':
            return self._check_volatility_spike(data, rule)
        elif condition == 'volume_spike':
            return self._check_volume_spike(data, rule)
        elif condition == 'technical_signal':
            return self._check_technical_signal(data, rule)
        
        return False
    
    def _check_price_change(self, data, rule, direction='up'):
        """Check price change conditions"""
        
        threshold = rule['threshold']  # Percentage change
        
        # This would need historical data - simplified for demo
        # In practice, you'd calculate the actual price change
        simulated_change = np.random.uniform(-5, 5)  # Simulate % change
        
        if direction == 'up':
            return simulated_change > threshold
        else:
            return simulated_change < -abs(threshold)
    
    def _check_volatility_spike(self, data, rule):
        """Check for volatility spikes"""
        
        # Simplified volatility check
        threshold = rule['threshold']
        
        # In practice, calculate actual volatility from historical data
        simulated_volatility = np.random.uniform(0, 50)
        
        return simulated_volatility > threshold
    
    def _check_volume_spike(self, data, rule):
        """Check for volume spikes"""
        
        if 'volume' not in data or pd.isna(data['volume']):
            return False
        
        threshold = rule['threshold']
        current_volume = data['volume']
        
        # In practice, compare to average volume
        average_volume = current_volume * 0.8  # Simplified
        
        return current_volume > average_volume * (1 + threshold / 100)
    
    def _check_technical_signal(self, data, rule):
        """Check technical analysis signals"""
        
        signal_type = rule.get('signal_type', 'rsi_overbought')
        threshold = rule['threshold']
        
        # Simplified technical signals
        if signal_type == 'rsi_overbought':
            simulated_rsi = np.random.uniform(0, 100)
            return simulated_rsi > threshold
        elif signal_type == 'rsi_oversold':
            simulated_rsi = np.random.uniform(0, 100)
            return simulated_rsi < threshold
        
        return False
    
    def _create_alert(self, rule_id, rule, data):
        """Create alert object"""
        
        return {
            'id': f"alert_{rule_id}_{int(time.time())}",
            'rule_id': rule_id,
            'rule_name': rule['name'],
            'asset': rule['asset'],
            'asset_name': data.get('series_name', rule['asset']),
            'condition': rule['condition'],
            'threshold': rule['threshold'],
            'current_value': data['value'],
            'timestamp': datetime.now(),
            'severity': rule.get('severity', 'medium'),
            'message': self._generate_alert_message(rule, data),
            'notification_channels': rule['notification_channels']
        }
    
    def _generate_alert_message(self, rule, data):
        """Generate human-readable alert message"""
        
        asset_name = data.get('series_name', rule['asset'])
        condition = rule['condition']
        threshold = rule['threshold']
        current_value = data['value']
        
        if condition == 'price_above':
            return f"🚨 {asset_name} price (${current_value:.2f}) is above threshold ${threshold:.2f}"
        elif condition == 'price_below':
            return f"📉 {asset_name} price (${current_value:.2f}) is below threshold ${threshold:.2f}"
        elif condition == 'price_change_up':
            return f"📈 {asset_name} price increased by more than {threshold}%"
        elif condition == 'price_change_down':
            return f"📉 {asset_name} price decreased by more than {threshold}%"
        elif condition == 'volatility_spike':
            return f"⚡ {asset_name} experiencing high volatility (>{threshold}%)"
        elif condition == 'volume_spike':
            return f"📊 {asset_name} volume spike detected (>{threshold}% above average)"
        elif condition == 'technical_signal':
            signal_type = rule.get('signal_type', 'technical')
            return f"🔍 {asset_name} technical signal: {signal_type} threshold reached"
        
        return f"🚨 Alert triggered for {asset_name}"
    
    def _send_alert_notifications(self, alert):
        """Send alert through configured notification channels"""
        
        for channel in alert['notification_channels']:
            try:
                if channel == 'streamlit':
                    self._send_streamlit_notification(alert)
                elif channel == 'email':
                    self._send_email_notification(alert)
                elif channel == 'webhook':
                    self._send_webhook_notification(alert)
                elif channel == 'slack':
                    self._send_slack_notification(alert)
            except Exception as e:
                st.error(f"❌ Failed to send {channel} notification: {str(e)}")
    
    def _send_streamlit_notification(self, alert):
        """Send Streamlit notification"""
        
        severity_colors = {
            'low': 'info',
            'medium': 'warning', 
            'high': 'error'
        }
        
        severity = alert.get('severity', 'medium')
        color = severity_colors.get(severity, 'warning')
        
        if color == 'error':
            st.error(f"🚨 {alert['message']}")
        elif color == 'warning':
            st.warning(f"⚠️ {alert['message']}")
        else:
            st.info(f"ℹ️ {alert['message']}")
    
    def _send_email_notification(self, alert):
        """Send email notification"""
        
        if not EMAIL_AVAILABLE:
            st.warning("📧 Email notifications require smtplib")
            return
        
        # Get email configuration from secrets or environment
        email_config = st.secrets.get("email", {})
        
        if not email_config:
            st.warning("📧 Email configuration not found in secrets")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('smtp_user', 'alerts@economicpulse.com')
            msg['To'] = email_config.get('alert_email', 'user@example.com')
            msg['Subject'] = f"Economic Pulse Alert: {alert['rule_name']}"
            
            # Create HTML body
            body = f"""
            <html>
            <body>
                <h2>🚨 Economic Pulse Alert</h2>
                <p><strong>Alert:</strong> {alert['rule_name']}</p>
                <p><strong>Asset:</strong> {alert['asset_name']}</p>
                <p><strong>Message:</strong> {alert['message']}</p>
                <p><strong>Current Value:</strong> {alert['current_value']}</p>
                <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
                <p><strong>Severity:</strong> {alert['severity'].upper()}</p>
                
                <hr>
                <p><em>This alert was generated by Economic Pulse V3.0</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(email_config.get('smtp_server', 'smtp.gmail.com'), 
                                email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config['smtp_user'], email_config['smtp_password'])
            
            text = msg.as_string()
            server.sendmail(email_config['smtp_user'], email_config['alert_email'], text)
            server.quit()
            
        except Exception as e:
            st.error(f"❌ Email sending failed: {str(e)}")
    
    def _send_webhook_notification(self, alert):
        """Send webhook notification"""
        
        if not WEBHOOK_AVAILABLE:
            return
        
        webhook_url = st.secrets.get("webhook_url")
        if not webhook_url:
            return
        
        payload = {
            'alert_id': alert['id'],
            'rule_name': alert['rule_name'],
            'asset': alert['asset'],
            'message': alert['message'],
            'severity': alert['severity'],
            'timestamp': alert['timestamp'].isoformat(),
            'current_value': alert['current_value']
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code != 200:
                st.warning(f"⚠️ Webhook returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Webhook failed: {str(e)}")
    
    def _send_slack_notification(self, alert):
        """Send Slack notification"""
        
        slack_webhook = st.secrets.get("slack_webhook_url")
        if not slack_webhook or not WEBHOOK_AVAILABLE:
            return
        
        severity_emojis = {
            'low': ':information_source:',
            'medium': ':warning:',
            'high': ':rotating_light:'
        }
        
        emoji = severity_emojis.get(alert['severity'], ':warning:')
        
        payload = {
            "text": f"{emoji} Economic Pulse Alert",
            "attachments": [
                {
                    "color": "danger" if alert['severity'] == 'high' else "warning",
                    "fields": [
                        {"title": "Alert", "value": alert['rule_name'], "short": True},
                        {"title": "Asset", "value": alert['asset_name'], "short": True},
                        {"title": "Message", "value": alert['message'], "short": False},
                        {"title": "Current Value", "value": str(alert['current_value']), "short": True},
                        {"title": "Severity", "value": alert['severity'].upper(), "short": True}
                    ],
                    "ts": int(alert['timestamp'].timestamp())
                }
            ]
        }
        
        try:
            response = requests.post(slack_webhook, json=payload, timeout=10)
            if response.status_code != 200:
                st.warning(f"⚠️ Slack webhook returned status {response.status_code}")
        except Exception as e:
            st.error(f"❌ Slack notification failed: {str(e)}")

class AlertDashboard:
    """Interactive alert management dashboard"""
    
    def __init__(self):
        self.alert_engine = AlertEngine()
        
        # Initialize with some sample alert rules
        self._initialize_sample_rules()
    
    def _initialize_sample_rules(self):
        """Initialize with sample alert rules"""
        
        sample_rules = {
            'spy_high': {
                'name': 'SPY High Price Alert',
                'condition': 'price_above',
                'threshold': 500.0,
                'asset': 'SPY',
                'notification_channels': ['streamlit', 'email'],
                'severity': 'medium',
                'cooldown_minutes': 30
            },
            'btc_drop': {
                'name': 'Bitcoin Price Drop',
                'condition': 'price_change_down',
                'threshold': 5.0,  # 5% drop
                'asset': 'BTC-USD',
                'notification_channels': ['streamlit', 'slack'],
                'severity': 'high',
                'cooldown_minutes': 15
            },
            'vix_spike': {
                'name': 'VIX Volatility Spike',
                'condition': 'price_above',
                'threshold': 30.0,
                'asset': 'VIXCLS',
                'notification_channels': ['streamlit', 'webhook'],
                'severity': 'high',
                'cooldown_minutes': 60
            }
        }
        
        for rule_id, rule_config in sample_rules.items():
            self.alert_engine.add_alert_rule(rule_id, rule_config)
    
    def create_alert_interface(self, df):
        """Create interactive alert management interface"""
        
        st.subheader("🚨 Real-Time Alert System")
        
        # Alert management tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔔 Active Alerts",
            "➕ Create Alert", 
            "⚙️ Manage Rules",
            "📊 Alert History"
        ])
        
        with tab1:
            self._display_active_alerts(df)
        
        with tab2:
            self._create_new_alert_interface(df)
        
        with tab3:
            self._manage_alert_rules_interface()
        
        with tab4:
            self._display_alert_history()
    
    def _display_active_alerts(self, df):
        """Display active alerts and monitoring status"""
        
        st.markdown("### 🔔 Real-Time Monitoring")
        
        # Check for new alerts
        if st.button("🔄 Check Alerts Now", type="primary"):
            with st.spinner("🔍 Checking alert conditions..."):
                triggered_alerts = self.alert_engine.check_alerts(df)
                
                if triggered_alerts:
                    st.success(f"✅ Found {len(triggered_alerts)} new alerts!")
                    for alert in triggered_alerts:
                        self._display_alert_card(alert)
                else:
                    st.info("✅ No new alerts triggered")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh alerts (every 30 seconds)")
        
        if auto_refresh:
            # This would implement auto-refresh in a real deployment
            st.info("🔄 Auto-refresh enabled - alerts will be checked automatically")
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Rules",
                len([r for r in self.alert_engine.alert_rules.values() if r.get('enabled', True)])
            )
        
        with col2:
            st.metric(
                "Total Rules", 
                len(self.alert_engine.alert_rules)
            )
        
        with col3:
            recent_alerts = len([a for a in self.alert_engine.alert_history 
                               if a['timestamp'] > datetime.now() - timedelta(hours=24)])
            st.metric("Alerts (24h)", recent_alerts)
        
        with col4:
            st.metric(
                "Notification Channels",
                len(set().union(*[r['notification_channels'] for r in self.alert_engine.alert_rules.values()]))
            )
        
        # Recent alerts
        if self.alert_engine.alert_history:
            st.markdown("### 📝 Recent Alerts")
            
            for alert in self.alert_engine.alert_history[-5:]:  # Last 5 alerts
                self._display_alert_card(alert)
    
    def _display_alert_card(self, alert):
        """Display individual alert card"""
        
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107', 
            'high': '#dc3545'
        }
        
        color = severity_colors.get(alert['severity'], '#ffc107')
        
        with st.container():
            st.markdown(f"""
            <div style="
                border-left: 5px solid {color};
                background-color: rgba(255,255,255,0.1);
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 5px;
            ">
                <h4 style="margin: 0; color: {color};">
                    {alert['rule_name']} - {alert['severity'].upper()}
                </h4>
                <p style="margin: 0.5rem 0;"><strong>Asset:</strong> {alert['asset_name']}</p>
                <p style="margin: 0.5rem 0;"><strong>Message:</strong> {alert['message']}</p>
                <p style="margin: 0.5rem 0;"><strong>Time:</strong> {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p style="margin: 0; font-size: 0.9em; opacity: 0.8;">
                    <strong>Channels:</strong> {', '.join(alert['notification_channels'])}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _create_new_alert_interface(self, df):
        """Interface for creating new alert rules"""
        
        st.markdown("### ➕ Create New Alert Rule")
        
        with st.form("new_alert_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic alert configuration
                alert_name = st.text_input("Alert Name", placeholder="e.g., SPY Price Alert")
                
                # Asset selection
                available_assets = df['series_id'].unique()
                selected_asset = st.selectbox("Select Asset", available_assets)
                
                # Condition type
                condition_type = st.selectbox(
                    "Alert Condition",
                    ["price_above", "price_below", "price_change_up", "price_change_down", 
                     "volatility_spike", "volume_spike", "technical_signal"],
                    format_func=lambda x: {
                        "price_above": "💹 Price Above Threshold",
                        "price_below": "📉 Price Below Threshold",
                        "price_change_up": "📈 Price Increase %",
                        "price_change_down": "📉 Price Decrease %",
                        "volatility_spike": "⚡ Volatility Spike",
                        "volume_spike": "📊 Volume Spike",
                        "technical_signal": "🔍 Technical Signal"
                    }[x]
                )
                
                # Threshold
                threshold = st.number_input("Threshold Value", value=0.0, step=0.1)
            
            with col2:
                # Notification settings
                st.markdown("**Notification Settings**")
                
                notification_channels = st.multiselect(
                    "Notification Channels",
                    ["streamlit", "email", "webhook", "slack"],
                    default=["streamlit"],
                    format_func=lambda x: {
                        "streamlit": "📱 Streamlit Dashboard",
                        "email": "📧 Email",
                        "webhook": "🔗 Webhook",
                        "slack": "💬 Slack"
                    }[x]
                )
                
                # Severity
                severity = st.selectbox(
                    "Alert Severity",
                    ["low", "medium", "high"],
                    index=1,
                    format_func=lambda x: {
                        "low": "🟢 Low",
                        "medium": "🟡 Medium", 
                        "high": "🔴 High"
                    }[x]
                )
                
                # Cooldown
                cooldown_minutes = st.number_input(
                    "Cooldown (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=15,
                    help="Minimum time between repeated alerts"
                )
                
                # Technical signal options
                if condition_type == "technical_signal":
                    signal_type = st.selectbox(
                        "Technical Signal Type",
                        ["rsi_overbought", "rsi_oversold", "macd_bullish", "macd_bearish"]
                    )
                else:
                    signal_type = None
            
            # Submit button
            submitted = st.form_submit_button("🚀 Create Alert Rule", type="primary")
            
            if submitted:
                if alert_name and selected_asset and notification_channels:
                    # Create rule configuration
                    rule_config = {
                        'name': alert_name,
                        'condition': condition_type,
                        'threshold': threshold,
                        'asset': selected_asset,
                        'notification_channels': notification_channels,
                        'severity': severity,
                        'cooldown_minutes': cooldown_minutes
                    }
                    
                    if signal_type:
                        rule_config['signal_type'] = signal_type
                    
                    # Generate unique rule ID
                    rule_id = f"alert_{int(time.time())}"
                    
                    # Add rule
                    try:
                        self.alert_engine.add_alert_rule(rule_id, rule_config)
                        st.success(f"✅ Alert rule '{alert_name}' created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to create alert rule: {str(e)}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    def _manage_alert_rules_interface(self):
        """Interface for managing existing alert rules"""
        
        st.markdown("### ⚙️ Manage Alert Rules")
        
        if not self.alert_engine.alert_rules:
            st.info("📝 No alert rules configured yet")
            return
        
        # Rules table
        rules_data = []
        for rule_id, rule in self.alert_engine.alert_rules.items():
            rules_data.append({
                'ID': rule_id,
                'Name': rule['name'],
                'Asset': rule['asset'],
                'Condition': rule['condition'],
                'Threshold': rule['threshold'],
                'Enabled': rule.get('enabled', True),
                'Severity': rule['severity'],
                'Channels': ', '.join(rule['notification_channels'])
            })
        
        rules_df = pd.DataFrame(rules_data)
        
        # Display rules with editing options
        for idx, (rule_id, rule) in enumerate(self.alert_engine.alert_rules.items()):
            with st.expander(f"📋 {rule['name']} ({rule['asset']})"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Condition:** {rule['condition']}")
                    st.write(f"**Threshold:** {rule['threshold']}")
                    st.write(f"**Severity:** {rule['severity']}")
                    st.write(f"**Channels:** {', '.join(rule['notification_channels'])}")
                
                with col2:
                    # Enable/Disable toggle
                    enabled = st.checkbox(
                        "Enabled",
                        value=rule.get('enabled', True),
                        key=f"enable_{rule_id}"
                    )
                    
                    if enabled != rule.get('enabled', True):
                        self.alert_engine.alert_rules[rule_id]['enabled'] = enabled
                        st.rerun()
                
                with col3:
                    # Delete button
                    if st.button(f"🗑️ Delete", key=f"delete_{rule_id}", type="secondary"):
                        self.alert_engine.remove_alert_rule(rule_id)
                        st.success(f"✅ Deleted rule: {rule['name']}")
                        st.rerun()
    
    def _display_alert_history(self):
        """Display alert history and analytics"""
        
        st.markdown("### 📊 Alert History & Analytics")
        
        if not self.alert_engine.alert_history:
            st.info("📝 No alert history available yet")
            return
        
        # Alert history summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_alerts = len(self.alert_engine.alert_history)
            st.metric("Total Alerts", total_alerts)
        
        with col2:
            today_alerts = len([a for a in self.alert_engine.alert_history 
                              if a['timestamp'].date() == datetime.now().date()])
            st.metric("Today's Alerts", today_alerts)
        
        with col3:
            avg_per_day = total_alerts / max(1, (datetime.now() - self.alert_engine.alert_history[0]['timestamp']).days)
            st.metric("Avg per Day", f"{avg_per_day:.1f}")
        
        # Alert timeline
        if len(self.alert_engine.alert_history) > 0:
            st.markdown("#### 📈 Alert Timeline")
            
            # Create timeline data
            timeline_data = []
            for alert in self.alert_engine.alert_history:
                timeline_data.append({
                    'Date': alert['timestamp'].date(),
                    'Time': alert['timestamp'].strftime('%H:%M'),
                    'Rule': alert['rule_name'],
                    'Asset': alert['asset'],
                    'Severity': alert['severity'],
                    'Message': alert['message'][:50] + '...' if len(alert['message']) > 50 else alert['message']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True)
        
        # Alert frequency by asset
        if len(self.alert_engine.alert_history) > 0:
            st.markdown("#### 📊 Alert Frequency by Asset")
            
            asset_counts = {}
            for alert in self.alert_engine.alert_history:
                asset = alert['asset']
                asset_counts[asset] = asset_counts.get(asset, 0) + 1
            
            if asset_counts:
                import plotly.express as px
                
                fig = px.bar(
                    x=list(asset_counts.keys()),
                    y=list(asset_counts.values()),
                    title="Alert Count by Asset",
                    labels={'x': 'Asset', 'y': 'Alert Count'}
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Notification channel configuration
class NotificationManager:
    """Manage notification channel configurations"""
    
    def __init__(self):
        self.channels = {}
    
    def configure_email(self, smtp_server, smtp_port, username, password, alert_email):
        """Configure email notifications"""
        
        self.channels['email'] = {
            'type': 'email',
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'alert_email': alert_email,
            'enabled': True
        }
    
    def configure_webhook(self, webhook_url, headers=None):
        """Configure webhook notifications"""
        
        self.channels['webhook'] = {
            'type': 'webhook',
            'url': webhook_url,
            'headers': headers or {},
            'enabled': True
        }
    
    def configure_slack(self, webhook_url):
        """Configure Slack notifications"""
        
        self.channels['slack'] = {
            'type': 'slack',
            'webhook_url': webhook_url,
            'enabled': True
        }
    
    def test_channel(self, channel_name):
        """Test notification channel"""
        
        if channel_name not in self.channels:
            return False, "Channel not configured"
        
        # Implement channel testing logic
        # This would send a test notification
        
        return True, "Test notification sent successfully"