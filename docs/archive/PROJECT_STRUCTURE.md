# 📂 Airbnb Price Predictor - Complete Project Structure & File Dictionary

This document provides a comprehensive overview of every file in the project, its purpose, current status, and whether it's needed for production.

## 🎯 **CORE PRODUCTION FILES** (Essential - Keep These)

### 🚀 **Main Application**
| File | Purpose | Status | Size |
|------|---------|--------|------|
| `streamlit_app.py` | **Main Streamlit web application** - Production UI | ✅ ACTIVE | Core |
| `code.ipynb` | **Model training notebook** - Contains the complete ML pipeline | ✅ ACTIVE | Core |

### 📊 **Data Files**
| File | Purpose | Status | Size |
|------|---------|--------|------|
| `listings.csv` | **Primary dataset** - Airbnb property features | ✅ ESSENTIAL | Large |
| `reviews.csv` | **Review text data** - Guest reviews for sentiment analysis | ✅ ESSENTIAL | Large |

### 🤖 **Model Files (JSON-based - Current Production)**
| File | Purpose | Status | Size |
|------|---------|--------|------|
| `model_data_for_streamlit.json` | **Primary model data** - Current production model | ✅ ACTIVE | Critical |
| `model_state.json` | **Model state backup** - Secondary model data | ✅ BACKUP | Medium |
| `training_data_export.json` | **Training data export** - For model recreation | ✅ ACTIVE | Medium |

### 🛠️ **Setup & Dependencies**
| File | Purpose | Status | Size |
|------|---------|--------|------|
| `requirements.txt` | **Python dependencies** - Package requirements | ✅ ESSENTIAL | Small |
| `setup.sh` | **Auto-setup script** - One-command installation | ✅ USEFUL | Small |
| `setup.py` | **Python setup utilities** - Environment setup | ✅ USEFUL | Small |

### 📚 **Documentation**
| File | Purpose | Status | Size |
|------|---------|--------|------|
| `README.md` | **Main project documentation** - Setup and usage guide | ✅ ESSENTIAL | Medium |
| `TROUBLESHOOTING.md` | **Debug guide** - Common issues and solutions | ✅ USEFUL | Medium |

---

## 🧪 **DEVELOPMENT & TESTING FILES** (Can Be Organized/Removed)

### ✅ **Useful Test/Demo Files** (Keep for Reference)
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `demo.py` | **Model demonstration** - Shows how to use trained models | 🔧 USEFUL | Keep |
| `test_prediction.py` | **Prediction testing** - Validates model functionality | 🔧 USEFUL | Keep |

### ⚠️ **Legacy/Debug Files** (Safe to Remove)
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `debug_shape_error.py` | Debug script for DataFrame shape issues | 🗑️ OBSOLETE | **REMOVE** |
| `test_dataframe.py` | DataFrame testing (issue resolved) | 🗑️ OBSOLETE | **REMOVE** |
| `test_dataframe_fix.py` | DataFrame fix testing | 🗑️ OBSOLETE | **REMOVE** |
| `test_explanation.py` | SHAP explanation testing | 🗑️ OBSOLETE | **REMOVE** |
| `test_final_fix.py` | Final fix testing | 🗑️ OBSOLETE | **REMOVE** |
| `test_shape_fix.py` | Shape error fix testing | 🗑️ OBSOLETE | **REMOVE** |
| `test_clean_models.py` | Clean model testing | 🗑️ OBSOLETE | **REMOVE** |
| `streamlit_app_json.py` | Old JSON-based Streamlit version | 🗑️ OBSOLETE | **REMOVE** |

### 🗃️ **Legacy Model Files** (Outdated - Can Remove)
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `multimodal_airbnb_model.pkl` | Old pickle model (v1) | 🗑️ OBSOLETE | **REMOVE** |
| `multimodal_airbnb_model_v2.pkl` | Old pickle model (v2) | 🗑️ OBSOLETE | **REMOVE** |
| `multimodal_model_clean.pkl` | Clean pickle model (superseded) | 🗑️ OBSOLETE | **REMOVE** |
| `tabular_airbnb_model.pkl` | Old tabular model (v1) | 🗑️ OBSOLETE | **REMOVE** |
| `tabular_airbnb_model_v2.pkl` | Old tabular model (v2) | 🗑️ OBSOLETE | **REMOVE** |
| `tabular_model_clean.pkl` | Clean tabular model (superseded) | 🗑️ OBSOLETE | **REMOVE** |
| `preprocessor.pkl` | Old preprocessor (v1) | 🗑️ OBSOLETE | **REMOVE** |
| `preprocessor_clean.pkl` | Clean preprocessor (superseded) | 🗑️ OBSOLETE | **REMOVE** |
| `preprocessor_v2.pkl` | Old preprocessor (v2) | 🗑️ OBSOLETE | **REMOVE** |
| `preprocessor_simple.pkl` | Simple preprocessor (may keep for fallback) | ⚠️ LEGACY | Consider keeping |
| `metadata.pkl` | Old metadata (v1) | 🗑️ OBSOLETE | **REMOVE** |
| `metadata_clean.pkl` | Clean metadata (superseded) | 🗑️ OBSOLETE | **REMOVE** |
| `metadata_v2.pkl` | Old metadata (v2) | 🗑️ OBSOLETE | **REMOVE** |

### 🗃️ **Legacy JSON Model Files** (Outdated)
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `streamlit_complete_model.json` | Old complete model JSON | 🗑️ OBSOLETE | **REMOVE** |
| `streamlit_linear_model.json` | Old linear model JSON | 🗑️ OBSOLETE | **REMOVE** |
| `streamlit_simple_model.json` | Old simple model JSON | 🗑️ OBSOLETE | **REMOVE** |

### 📁 **Model Artifacts Directory**
| Path | Purpose | Status | Action |
|------|---------|--------|--------|
| `model_artifacts/` | Contains joblib model files | ⚠️ LEGACY | May remove if JSON works |
| `├── metadata.joblib` | Joblib metadata | ⚠️ LEGACY | Backup option |
| `├── multimodal_model.joblib` | Joblib multimodal model | ⚠️ LEGACY | Backup option |
| `├── preprocessor.joblib` | Joblib preprocessor | ⚠️ LEGACY | Backup option |
| `└── tabular_model.joblib` | Joblib tabular model | ⚠️ LEGACY | Backup option |

### 📚 **Documentation Files**
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `NUMPY_FIX_SUMMARY.md` | NumPy compatibility fix documentation | 🔧 USEFUL | Keep for reference |

### 🏗️ **System Files**
| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `.venv/` | Python virtual environment | ✅ ESSENTIAL | Keep (active environment) |
| `__pycache__/` | Python cache files | 🗑️ CACHE | Auto-generated (can ignore) |

---

## 🧹 **CLEANUP RECOMMENDATIONS**

### 🚨 **Safe to Remove Immediately** (26 files)
```bash
# Debug and test files (no longer needed)
rm debug_shape_error.py
rm test_dataframe.py test_dataframe_fix.py test_explanation.py 
rm test_final_fix.py test_shape_fix.py test_clean_models.py
rm streamlit_app_json.py

# Legacy pickle models (superseded by JSON)
rm multimodal_airbnb_model.pkl multimodal_airbnb_model_v2.pkl multimodal_model_clean.pkl
rm tabular_airbnb_model.pkl tabular_airbnb_model_v2.pkl tabular_model_clean.pkl
rm preprocessor.pkl preprocessor_clean.pkl preprocessor_v2.pkl
rm metadata.pkl metadata_clean.pkl metadata_v2.pkl

# Legacy JSON models (superseded)
rm streamlit_complete_model.json streamlit_linear_model.json streamlit_simple_model.json
```

### ⚠️ **Consider Removing** (with backup)
```bash
# Create backup first
mkdir backup_model_artifacts
cp -r model_artifacts/ backup_model_artifacts/

# Then optionally remove if JSON models work perfectly
# rm -r model_artifacts/
```

### ✅ **Keep These** (12 essential files)
```
streamlit_app.py              # Main application
code.ipynb                    # Model training
listings.csv                  # Data
reviews.csv                   # Data
model_data_for_streamlit.json # Current model
model_state.json              # Backup model
training_data_export.json     # Training data
requirements.txt              # Dependencies
setup.sh                      # Setup script
setup.py                      # Setup utilities
README.md                     # Documentation
TROUBLESHOOTING.md            # Debug guide
```

---

## 🎯 **FINAL PRODUCTION STRUCTURE** (After Cleanup)

```
📂 airbnb-price-predictor/
├── 🚀 **CORE APPLICATION**
│   ├── streamlit_app.py          # Main Streamlit web app
│   └── code.ipynb                # Model training notebook
├── 📊 **DATA**
│   ├── listings.csv              # Property features dataset
│   └── reviews.csv               # Guest reviews dataset
├── 🤖 **MODELS** (JSON-based)
│   ├── model_data_for_streamlit.json  # Primary production model
│   ├── model_state.json               # Backup model state
│   └── training_data_export.json      # Training data export
├── 🛠️ **SETUP**
│   ├── requirements.txt          # Python dependencies
│   ├── setup.sh                  # Auto-setup script
│   └── setup.py                  # Setup utilities
├── 📚 **DOCUMENTATION**
│   ├── README.md                 # Main documentation
│   ├── TROUBLESHOOTING.md        # Debug guide
│   ├── NUMPY_FIX_SUMMARY.md      # NumPy fix documentation
│   └── PROJECT_STRUCTURE.md      # This file
├── 🧪 **DEVELOPMENT** (Optional)
│   ├── demo.py                   # Model demonstration
│   └── test_prediction.py        # Prediction testing
└── 🏗️ **SYSTEM**
    ├── .venv/                    # Virtual environment
    └── __pycache__/              # Python cache (auto-generated)
```

---

## 📋 **FILE SIZE & CLEANUP IMPACT**

| Category | Files | Estimated Space Saved |
|----------|-------|----------------------|
| Legacy Pickle Models | 12 files | ~500-800 MB |
| Debug/Test Scripts | 8 files | ~50-100 KB |
| Legacy JSON Models | 3 files | ~10-50 MB |
| **Total Cleanup** | **23 files** | **~550-850 MB** |

---

## 🚀 **QUICK START AFTER CLEANUP**

```bash
# 1. Essential files verification
ls streamlit_app.py code.ipynb listings.csv reviews.csv requirements.txt

# 2. Run the application
streamlit run streamlit_app.py

# 3. Model training (if needed)
jupyter notebook code.ipynb
```

This structure maintains all essential functionality while removing outdated and redundant files, making the project cleaner and easier to maintain.
