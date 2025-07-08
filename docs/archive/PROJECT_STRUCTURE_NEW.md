# 📂 Airbnb Smart Pricing Engine - Project Structure

This document provides a comprehensive overview of the organized project structure after cleanup and categorization.

## 🎯 **PROJECT OVERVIEW**

The project has been organized into a clean, maintainable structure following industry best practices. All unnecessary files have been removed and backed up, and remaining files are categorized by function.

---

## 📁 **DIRECTORY STRUCTURE**

### 🚀 **`src/` - Main Application Code**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `streamlit_app.py` | **Primary Web Application** | ✅ PRODUCTION | Beautiful Streamlit app with modern UI, explainable AI, and interactive features |
| `demo.py` | **Model Demonstration** | 🔧 DEVELOPMENT | Shows how to use trained models programmatically |

### 📓 **`notebooks/` - Jupyter Notebooks**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `code.ipynb` | **Complete ML Pipeline** | ✅ ACTIVE | Model training, evaluation, feature engineering, and analysis |

### 📊 **`data/` - Dataset Files**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `listings.csv` | **Property Features Dataset** | ✅ ESSENTIAL | Primary Airbnb property data with 50+ features |
| `reviews.csv` | **Guest Reviews Dataset** | ✅ ESSENTIAL | Text data for sentiment analysis and NLP features |

### 🤖 **`models/` - Model Artifacts**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `model_data_for_streamlit.json` | **Production Model Data** | ✅ PRODUCTION | JSON-serialized model for Streamlit deployment |
| `model_state.json` | **Model State Backup** | ✅ BACKUP | Secondary model data and metadata |
| `training_data_export.json` | **Training Data Export** | ✅ ACTIVE | Exported training data for model recreation |
| `preprocessor_simple.pkl` | **Data Preprocessor** | ✅ ACTIVE | Sklearn preprocessor for data transformation |

### 📚 **`docs/` - Documentation**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `README.md` | **Main Documentation** | ✅ ESSENTIAL | Project overview, setup, and usage guide |
| `PROJECT_STRUCTURE.md` | **File Organization** | ✅ ESSENTIAL | This file - complete project structure |
| `TROUBLESHOOTING.md` | **Debug Guide** | ✅ USEFUL | Common issues, solutions, and debugging tips |
| `PROJECT_CLEANUP_SUMMARY.md` | **Cleanup History** | 📋 REFERENCE | Record of project cleanup and organization |
| `NUMPY_FIX_SUMMARY.md` | **Technical Fixes** | 📋 REFERENCE | Documentation of technical issue resolutions |
| `FILE_DICTIONARY_COMPLETE.md` | **Legacy File Reference** | 📋 ARCHIVE | Complete file dictionary from before cleanup |
| `README_COMPLETE.md` | **Extended Documentation** | 📋 REFERENCE | Comprehensive project documentation |

### 🛠️ **`scripts/` - Setup & Utilities**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `setup.sh` | **One-Command Setup** | ✅ ESSENTIAL | Automated environment setup and dependency installation |
| `setup.py` | **Python Setup Utilities** | ✅ USEFUL | Python environment configuration helpers |
| `cleanup_project.sh` | **Cleanup Utilities** | 🔧 MAINTENANCE | Script used for project organization (archived) |

### 🧪 **`tests/` - Test Files**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `test_prediction.py` | **Model Validation** | ✅ USEFUL | Tests for model functionality and predictions |

### ⚙️ **`config/` - Configuration**
| File | Purpose | Status | Description |
|------|---------|--------|-------------|
| `requirements.txt` | **Python Dependencies** | ✅ ESSENTIAL | Complete list of required Python packages |

### 💾 **`backup/` - Backup Files**
| Directory | Purpose | Status | Description |
|-----------|---------|--------|-------------|
| `backup_removed_files_20250707_201707/` | **Cleanup Backup** | 📦 ARCHIVED | Complete backup of all files removed during cleanup |

---

## 🎨 **APPLICATION FEATURES**

### **Streamlit Web Application** (`src/streamlit_app.py`)
- **🎨 Modern UI**: Glassmorphism design with custom CSS styling
- **📊 Interactive Charts**: Plotly-powered visualizations for feature importance
- **🔍 Explainable AI**: SHAP-based model explanations
- **📈 Sensitivity Analysis**: Dynamic feature impact analysis
- **💡 Smart Recommendations**: AI-powered pricing optimization tips
- **📱 Responsive Design**: Mobile-friendly interface
- **🎯 Real-time Predictions**: Instant price predictions with explanations

### **ML Pipeline** (`notebooks/code.ipynb`)
- **🤖 Multimodal Learning**: Tabular + Text data fusion
- **🔧 Feature Engineering**: 50+ engineered features
- **📊 Model Ensemble**: Random Forest + Gradient Boosting + Extra Trees
- **🎯 BERT Integration**: DistilBERT for review sentiment analysis
- **📈 Performance Tracking**: Comprehensive evaluation metrics
- **🔍 Explainability**: SHAP value analysis for interpretability

---

## 📋 **FILE STATUS LEGEND**

| Status | Meaning | Action |
|--------|---------|--------|
| ✅ PRODUCTION | Currently used in production | Keep and maintain |
| ✅ ESSENTIAL | Critical for project function | Keep and maintain |
| ✅ ACTIVE | Actively used for development | Keep and maintain |
| ✅ USEFUL | Helpful for development/testing | Keep for now |
| 🔧 DEVELOPMENT | Development/testing only | Optional |
| 🔧 MAINTENANCE | Maintenance utilities | Archive after use |
| 📋 REFERENCE | Documentation/reference | Archive safely |
| 📦 ARCHIVED | Backed up, not in active use | Safe to remove |

---

## 🚀 **GETTING STARTED**

### **Quick Start**
1. **Setup**: Run `./scripts/setup.sh` for one-command installation
2. **Launch**: Execute `streamlit run src/streamlit_app.py`
3. **Explore**: Open `http://localhost:8501` in your browser

### **Development**
1. **Notebooks**: Open `notebooks/code.ipynb` for ML development
2. **Testing**: Run `python tests/test_prediction.py` for validation
3. **Documentation**: Refer to files in `docs/` for detailed info

---

## 🧹 **CLEANUP SUMMARY**

### **Files Removed** (24 total)
- ❌ **Debug Scripts**: 10 temporary debugging files
- ❌ **Legacy Models**: 12 old pickle/JSON model files  
- ❌ **Duplicate Files**: 2 redundant configuration files

### **Files Organized** (Current)
- ✅ **Core Files**: 12 essential production files
- ✅ **Documentation**: 7 comprehensive documentation files
- ✅ **Development**: 3 development and testing files

### **Backup Created**
- 📦 All removed files backed up in `backup/backup_removed_files_20250707_201707/`
- 🔒 Complete backup ensures no data loss
- 📅 Timestamped for easy identification

---

## 🎯 **NEXT STEPS**

1. **✅ COMPLETE**: Project cleanup and organization
2. **✅ COMPLETE**: Documentation updates  
3. **🔄 ONGOING**: Development using organized structure
4. **📊 FUTURE**: Deployment preparation using clean codebase
5. **🔧 FUTURE**: Additional feature development in organized folders

---

*Last Updated: July 7, 2025*  
*Project Status: Clean, Organized, Production-Ready* ✨
