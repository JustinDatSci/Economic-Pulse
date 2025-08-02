# 🔑 API Configuration Guide - Economic Pulse V3.1

## Overview
Economic Pulse V3.1 supports multiple data sources for real-time financial data. While the application works with simulated data by default, configuring real APIs provides access to live market data and enhanced functionality.

## Supported APIs

### 1. 📊 FRED (Federal Reserve Economic Data) - **Recommended**
- **Purpose**: US economic indicators (unemployment, inflation, GDP, etc.)
- **Cost**: FREE
- **Rate Limits**: 120 requests/minute
- **Data Quality**: Excellent (official government data)

#### Setup Instructions:
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill out the simple form (name, email, intended use)
4. Receive your API key via email (usually instant)
5. Add to Streamlit secrets or environment variables

#### Configuration:
```toml
# .streamlit/secrets.toml
[secrets]
FRED_API_KEY = "your_fred_api_key_here"
```

### 2. 📈 Alpha Vantage - **Optional**
- **Purpose**: Stock prices, forex, crypto data
- **Cost**: FREE tier (5 requests/minute, 500 requests/day)
- **Rate Limits**: Very restrictive on free tier
- **Data Quality**: Good

#### Setup Instructions:
1. Visit: https://www.alphavantage.co/support/#api-key
2. Enter your email address
3. Receive API key instantly
4. Add to configuration

#### Configuration:
```toml
# .streamlit/secrets.toml
[secrets]
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"
```

### 3. 📊 Yahoo Finance (via yfinance) - **Automatic**
- **Purpose**: Stock prices, crypto, forex data
- **Cost**: FREE
- **Rate Limits**: Reasonable (no official limits)
- **Data Quality**: Good
- **Setup**: No API key required (works automatically)

## Configuration Methods

### Method 1: Streamlit Secrets (Recommended for Streamlit Cloud)
Create `.streamlit/secrets.toml` in your project directory:

```toml
[secrets]
FRED_API_KEY = "your_fred_api_key_here"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"

[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_user = "your_email@gmail.com"
smtp_password = "your_app_password"
alert_email = "alerts@yourdomain.com"

[notifications]
slack_webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
webhook_url = "https://your-webhook-endpoint.com/alerts"
```

### Method 2: Environment Variables (For local development)
Add to your `.env` file or system environment:

```bash
# .env file
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Method 3: Direct Configuration (In-app settings)
Use the Settings page in the application to configure APIs directly.

## Email Notifications Setup

### Gmail Configuration:
1. Enable 2-factor authentication on your Gmail account
2. Generate an "App Password":
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use this app password (not your regular Gmail password)

### Configuration:
```toml
[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_user = "your_email@gmail.com"
smtp_password = "your_16_character_app_password"
alert_email = "alerts@yourdomain.com"
```

## Slack Notifications Setup

1. Create a Slack app at: https://api.slack.com/apps
2. Add "Incoming Webhooks" feature
3. Create webhook for your desired channel
4. Copy webhook URL to configuration

## Webhook Notifications Setup

Configure custom webhook endpoints for integration with other systems:

```toml
[notifications]
webhook_url = "https://your-endpoint.com/economic-pulse-alerts"
```

## Testing Your Configuration

### 1. Check API Status
The application automatically tests API connectivity and displays status in the sidebar:
- 🟢 Online: API is working
- 🔴 Offline: API key missing or invalid

### 2. Test Notifications
Use the Settings page "Test Notifications" feature to verify email/Slack setup.

### 3. Verify Data Sources
Check the dashboard for real vs simulated data indicators.

## Rate Limits and Best Practices

### FRED API:
- ✅ 120 requests/minute
- ✅ Very generous limits
- ✅ Perfect for economic indicators

### Alpha Vantage (Free):
- ⚠️ 5 requests/minute
- ⚠️ 500 requests/day
- ⚠️ Consider paid plan for frequent use

### Yahoo Finance:
- ✅ No official rate limits
- ✅ Be respectful with request frequency
- ✅ Built-in error handling

## Fallback Behavior

Economic Pulse V3.1 is designed to work gracefully without API keys:

1. **No API Keys**: Uses realistic simulated data
2. **Partial Configuration**: Uses real data where possible, simulated elsewhere
3. **API Failures**: Automatically falls back to cached or simulated data
4. **Rate Limiting**: Implements intelligent caching and request throttling

## Security Best Practices

### ✅ DO:
- Use Streamlit secrets for production deployments
- Use environment variables for local development
- Enable 2FA on all accounts
- Use app passwords instead of main passwords
- Regularly rotate API keys

### ❌ DON'T:
- Commit API keys to version control
- Share API keys in public forums
- Use production keys in development
- Hardcode credentials in source code

## Troubleshooting

### Common Issues:

1. **"API key not found"**
   - Check file path: `.streamlit/secrets.toml`
   - Verify key name matches exactly
   - Restart Streamlit after adding keys

2. **"Rate limit exceeded"**
   - Wait for rate limit reset
   - Reduce refresh frequency
   - Consider upgrading to paid tier

3. **"Email notifications not working"**
   - Verify app password (not regular password)
   - Check 2FA is enabled
   - Test with Gmail directly first

4. **"Connection errors"**
   - Check internet connectivity
   - Verify API endpoints are accessible
   - Check firewall/proxy settings

## Getting Help

If you encounter issues:

1. Check the application logs in Streamlit
2. Verify API status on provider websites
3. Test configuration step by step
4. Check the troubleshooting section above

## Advanced Configuration

### Custom Data Sources
The application is designed to be extensible. You can add custom data sources by:

1. Creating a new loader class in `enhanced_data_loader.py`
2. Adding API configuration options
3. Implementing error handling and fallbacks

### Performance Optimization
- Enable caching for frequently accessed data
- Configure appropriate refresh intervals
- Use data compression for large datasets
- Implement request batching where possible

---

## Quick Start Checklist

- [ ] Get FRED API key (recommended)
- [ ] Create `.streamlit/secrets.toml` file
- [ ] Add API keys to secrets file
- [ ] Test configuration in Settings page
- [ ] Verify real data is loading
- [ ] Configure email notifications (optional)
- [ ] Set up Slack notifications (optional)
- [ ] Test alert system

**Estimated setup time: 10-15 minutes**

---

*For additional support or questions about API configuration, please refer to the application's Settings page or check the individual API provider documentation.*