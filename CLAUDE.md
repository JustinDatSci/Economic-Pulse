# 🤖 Claude Code - Economic Pulse Project Memory

## 📊 Project Context
**Economic Pulse V3.1** - Advanced Financial Intelligence Platform
- **Tech Stack**: Streamlit, TensorFlow, scikit-learn, Plotly, pandas
- **Architecture**: Modular components with real-time data integration
- **APIs**: FRED (economic), Alpha Vantage (market), Yahoo Finance (backup)
- **Features**: LSTM predictions, portfolio optimization, alert system, modern UI

## 🚀 Key Commands & Shortcuts

### Quick Development Commands
```bash
# Start development server
streamlit run app.py

# Run with specific requirements
pip install -r requirements_v3_1.txt

# Test API connections
python -c "import streamlit as st; print('Streamlit ready')"

# Git workflow
git add . && git commit -m "feature: description" && git push origin main
```

### Essential File References
- `@app.py` - Main V3.1 application
- `@requirements_v3_1.txt` - Complete dependencies
- `@.streamlit/secrets.toml` - API configuration
- `@enhanced_data_loader.py` - Real-time data
- `@advanced_ml_models.py` - LSTM & ML models
- `@portfolio_optimizer.py` - Portfolio tools
- `@alert_system.py` - Notification system

## 🛠 Common Workflow Patterns

### Feature Development
1. **Plan**: Use TodoWrite for multi-step features
2. **Research**: Reference existing code with @files
3. **Implement**: Batch related changes together
4. **Test**: Verify with `streamlit run app.py`
5. **Deploy**: Commit and push for Streamlit Cloud auto-deploy

### Bug Fixes
1. **Identify**: Use extended thinking "think harder about this error"
2. **Locate**: Search with Task tool for complex code searches
3. **Fix**: Edit in place with context preservation
4. **Verify**: Test locally before committing

### API Integration
1. **Check**: Verify API status in app sidebar
2. **Configure**: Update `.streamlit/secrets.toml`
3. **Test**: Use Settings page test functions
4. **Fallback**: Ensure graceful degradation to simulated data

## 📁 Project Structure Memory
```
economic-pulse/
├── 🚀 app.py                          # Main V3.1 application (CURRENT)
├── 📊 enhanced_data_loader.py          # Real-time APIs
├── 🧠 advanced_ml_models.py            # LSTM & ensemble models  
├── 💼 portfolio_optimizer.py           # Modern Portfolio Theory
├── 🚨 alert_system.py                  # Multi-channel alerts
├── 🎨 enhanced_ui_components.py        # Modern UI components
├── 📋 requirements_v3_1.txt            # V3.1 dependencies
├── 🔑 API_SETUP_GUIDE.md               # API configuration guide
└── 📖 README.md                        # Platform documentation
```

## 🔧 Environment Setup
- **Python**: 3.9+ (recommended 3.10)
- **Main Dependencies**: streamlit, tensorflow, scikit-learn, plotly, cvxpy
- **Optional**: redis (caching), pydantic (validation)
- **APIs**: FRED (free), Alpha Vantage (5 req/min free), Yahoo Finance (auto)

## 🚨 Critical Workflows

### Deploy New Version
```bash
# 1. Update version in app.py header
# 2. Test locally: streamlit run app.py  
# 3. Commit: git add . && git commit -m "version: X.X updates"
# 4. Push: git push origin main
# 5. Verify: Check Streamlit Cloud auto-deployment
```

### API Issues
```bash
# Check API status: Look for 🟢/🔴 indicators in sidebar
# Test config: Use Settings page "Test Notifications" 
# Fallback: App works without APIs using simulated data
# Debug: Check .streamlit/secrets.toml format
```

### Performance Optimization
```bash
# Enable caching: Use @st.cache_data decorators
# Optimize requests: Batch API calls where possible
# Monitor: Check Streamlit Cloud metrics
# Scale: Consider upgrading API tiers for heavy usage
```

## 💡 Best Practices Learned
1. **Always use TodoWrite** for multi-step features (3+ tasks)
2. **Reference files with @** instead of describing them
3. **Batch related changes** together in single requests
4. **Test locally first** before committing to avoid broken deployments
5. **Use graceful fallbacks** for all external dependencies
6. **Commit frequently** with descriptive messages
7. **Keep API keys secure** - never commit to version control

## 🎯 Comprehensive Accuracy Framework
### **Pre-Work Validation:**
- ✅ Check environment context (date: 2025-08-02, platform: darwin)
- ✅ Verify file paths exist before referencing
- ✅ Read existing code patterns before editing
- ✅ Cross-check project version consistency (V3.1)
- ✅ Validate API endpoints and configurations

### **During Work:**
- ✅ Follow established code patterns and naming
- ✅ Use proper imports and dependencies
- ✅ Implement error handling and fallbacks
- ✅ Match existing project style and structure

### **Post-Work Review:**
- ✅ Scan for hardcoded dates/paths without verification
- ✅ Check consistency across all generated content
- ✅ Verify all file references and configurations
- ✅ Suggest appropriate testing and validation steps

## 🎯 Common Request Patterns

### Efficient Requests
```bash
# ✅ Good: "Update @advanced_ml_models.py to add ensemble learning, test in @app.py, update @requirements_v3_1.txt if needed"
# ❌ Avoid: "Can you look at the ML models file and see if we can add ensemble learning?"
```

### Multi-file Operations
```bash
# ✅ Good: "Add real-time crypto data to @enhanced_data_loader.py and integrate display in @app.py dashboard"
# ❌ Avoid: Step-by-step requests for each file separately
```

## 🔄 Deployment Pipeline
1. **Local Development** → `streamlit run app.py`
2. **Git Commit** → `git add . && git commit -m "..."`
3. **GitHub Push** → `git push origin main`
4. **Auto-Deploy** → Streamlit Cloud automatically deploys
5. **Verify** → Check https://economic-pulse.streamlit.app

## 👁️ Live Project Monitoring
### **Enable Claude Code to "see" live project:**
- **Screenshots**: May have technical limitations with some file types
- **Detailed descriptions**: Describe what you see, any errors, layout issues
- **Copy/paste text**: Error messages, logs, specific content
- **Specific questions**: "Does it show V3.1?" "Are charts loading?" "Any error messages?"
- **WebFetch**: Limited success with dynamic Streamlit apps

## 📚 External Resources
- **Live App**: https://economic-pulse.streamlit.app
- **GitHub**: https://github.com/JustinDatSci/economic-pulse
- **FRED API**: https://fred.stlouisfed.org/docs/api/
- **Alpha Vantage**: https://www.alphavantage.co/documentation/
- **Streamlit Docs**: https://docs.streamlit.io/

---
*Last Updated: August 2025 - V3.1 Complete*
*This file helps Claude Code understand your project context and preferred workflows*