# 🚀 Claude Code Workflow Templates

## 📋 Template 1: Financial/ML Project Setup

### Initial Project Structure
```bash
# Quick start for new financial analysis projects
project-name/
├── 🚀 app.py                    # Main Streamlit application
├── 📊 data_loader.py            # Data fetching and processing
├── 🧠 ml_models.py              # Machine learning models
├── 📈 analysis.py               # Statistical analysis
├── 🎨 ui_components.py          # Custom UI elements
├── 📋 requirements.txt          # Dependencies
├── 🔑 API_SETUP.md              # API configuration guide
├── 🤖 CLAUDE.md                 # Claude Code project memory
└── 📖 README.md                 # Project documentation
```

### Standard Dependencies
```txt
# Core requirements for financial ML projects
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
yfinance>=0.2.18
requests>=2.31.0
```

### Claude Code Setup Commands
```bash
# 1. Create project memory
echo "# Project Context: [Brief description]" > CLAUDE.md

# 2. Set up git
git init && git add . && git commit -m "initial: project setup"

# 3. Reference pattern for Claude Code
# Use @app.py @data_loader.py for quick file references
```

## 📋 Template 2: Feature Development

### Standard Feature Request Pattern
```bash
# ✅ Efficient pattern:
"Add [feature] to @[main_file].py, update dependencies in @requirements.txt, and test integration"

# ✅ For complex features:
"Plan and implement [feature]: research existing code, design architecture, implement, test, document"
```

### TodoWrite Integration
```bash
# Always use for multi-step features:
1. Research and design
2. Implement core functionality  
3. Add UI components
4. Test and validate
5. Update documentation
```

## 📋 Template 3: API Integration

### Standard API Addition
```bash
# File structure for new APIs:
1. Add API class to @data_loader.py
2. Update configuration in @API_SETUP.md
3. Add UI indicators in @app.py
4. Test fallback behavior
5. Update @requirements.txt if needed
```

### Security Checklist
```bash
# ✅ API key security:
- Store in .streamlit/secrets.toml (never commit)
- Document in API_SETUP.md (format only)
- Implement graceful fallbacks
- Add status indicators in UI
```

## 📋 Template 4: ML Model Integration

### Standard ML Pipeline
```bash
# Model development pattern:
1. Data preprocessing in @data_loader.py
2. Model architecture in @ml_models.py  
3. Training and validation logic
4. Prediction interface in @app.py
5. Performance monitoring dashboard
```

### Model Files Structure
```python
# @ml_models.py template:
class ModelName:
    def __init__(self): pass
    def prepare_data(self, data): pass
    def train(self, X, y): pass
    def predict(self, X): pass
    def evaluate(self, X, y): pass
```

## 📋 Template 5: UI/UX Development

### Streamlit UI Pattern
```python
# Standard page layout:
st.set_page_config(page_title="App Name", page_icon="🚀")
st.title("🚀 App Name - Description")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Settings")
    # Controls here

# Main content area
col1, col2 = st.columns([2, 1])
with col1:
    # Primary content
with col2:
    # Secondary info/metrics
```

### Custom Components Pattern
```python
# @ui_components.py template:
def custom_metric_card(title, value, delta=None):
    # Reusable metric display
    
def custom_chart(data, chart_type="line"):
    # Standardized chart formatting
    
def status_indicator(status, label):
    # API/service status display
```

## 📋 Template 6: Testing & Deployment

### Local Testing Checklist
```bash
# Before committing:
✅ streamlit run app.py (test locally)
✅ Check all API connections
✅ Verify fallback behavior
✅ Test with/without API keys
✅ Review UI responsiveness
```

### Deployment Commands
```bash
# Standard deployment flow:
git add .
git commit -m "[type]: [description]"
git push origin main
# Verify auto-deployment on hosting platform
```

## 📋 Template 7: Documentation

### README.md Template
```markdown
# 🚀 Project Name - Brief Description

## Features
- Feature 1
- Feature 2  
- Feature 3

## Quick Start
```bash
git clone [repo]
cd [project]
pip install -r requirements.txt
streamlit run app.py
```

## API Setup
See [API_SETUP.md](API_SETUP.md)
```

### API_SETUP.md Template
```markdown
# API Configuration Guide

## Required APIs
1. **API Name** - Purpose
   - Get key: [URL]
   - Rate limits: [Details]
   - Configuration: Add to .streamlit/secrets.toml

## Configuration Format
```toml
[secrets]
API_KEY_NAME = "your_key_here"
```
```

## 🎯 Quick Reference Commands

### File Operations
```bash
# Reference multiple files efficiently:
@app.py @data_loader.py @requirements.txt

# Search for patterns:
"Search for all API integrations in the codebase"

# Batch operations:
"Update API handling in @data_loader.py and add UI status in @app.py"
```

### Development Shortcuts
```bash
# Memory shortcuts:
#context: Financial ML platform with real-time data
#remember: Use graceful fallbacks for all APIs

# Extended thinking:
"think harder about the optimal architecture for this feature"

# Specialized agents:
"Use the general-purpose agent to research ML model implementations"
```

### Git Patterns
```bash
# Commit message patterns:
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in component"  
git commit -m "docs: update API setup guide"
git commit -m "refactor: improve code structure"
```

---

## 🚀 Future Project Initialization

### New Financial Project Setup
```bash
# 1. Clone or create repository
# 2. Copy WORKFLOW_TEMPLATES.md to new project
# 3. Create CLAUDE.md with project context
# 4. Set up standard file structure
# 5. Initialize with core dependencies
# 6. Document API requirements
# 7. Set up development workflow
```

### Adaptation Guidelines
- Customize file structure based on project needs
- Adjust dependencies for specific requirements  
- Modify API patterns for different data sources
- Adapt UI templates for project branding
- Scale TodoWrite usage based on complexity

---
*Templates updated: August 2025*
*Designed for efficient Claude Code workflows with financial/ML projects*