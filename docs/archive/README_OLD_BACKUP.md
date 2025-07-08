# 🏠 Airbnb Smart Pricing Engine

A comprehensive machine learning solution that predicts Airbnb prices using both tabular property data and guest review text, with built-in explainability features and an interactive web interface for property owners.

## 🌟 Key Features

- **🤖 Multimodal AI**: Combines property features with guest review sentiment using DistilBERT
- **🔍 Explainable AI**: SHAP-based explanations showing which features impact pricing
- **📊 Interactive UI**: Beautiful Streamlit web app with modern design
- **📈 Sensitivity Analysis**: Interactive charts showing how features affect price
- **💡 Actionable Insights**: Get specific recommendations for price optimization
- **🎯 Feature Importance**: Comprehensive analysis of what drives pricing decisions
- **📱 Responsive Design**: Mobile-friendly interface with glassmorphism styling

## 📊 Model Performance

- **Tabular Model**: Random Forest + Gradient Boosting + Extra Trees ensemble
- **Text Model**: DistilBERT embeddings for review sentiment analysis
- **Meta-learner**: Combines both models for final prediction
- **Evaluation**: Cross-validation with R² and MAE metrics
- **Improvement**: +0.9% R² improvement, +6.1% MAE improvement
- **Cross-Validation**: 85.1% ± 2.0% (excellent stability)
- **MAE**: $26.72 (vs $28.45 for tabular-only)

### Architecture Highlights
- **Multimodal fusion** of tabular and text data
- **DistilBERT** for review text encoding
- **Ensemble learning** with 3 optimized models
- **Advanced preprocessing** with power transforms
- **Production-ready** implementation

## 🚀 Quick Start

### 1. Setup (One Command)
```bash
./scripts/setup.sh
```

### 2. Run the Application
```bash
streamlit run src/streamlit_app.py
```

### 3. Open in Browser
The app will automatically open at `http://localhost:8501`

---

## 📁 Project Structure

```
📦 Airbnb Smart Pricing Engine
├── 📂 src/                    # Main application code
│   ├── streamlit_app.py       # Primary Streamlit web app
│   └── demo.py               # Model demonstration script
├── 📂 notebooks/             # Jupyter notebooks
│   └── code.ipynb           # Complete ML pipeline & training
├── 📂 data/                  # Dataset files
│   ├── listings.csv         # Property features (primary dataset)
│   └── reviews.csv          # Guest reviews for sentiment analysis
├── 📂 models/                # Model artifacts & data
│   ├── model_data_for_streamlit.json  # Production model data
│   ├── model_state.json               # Model state backup
│   ├── training_data_export.json      # Training data export
│   └── preprocessor_simple.pkl        # Data preprocessor
├── 📂 docs/                  # Documentation
│   ├── README.md            # This file
│   ├── PROJECT_STRUCTURE.md # Detailed file organization
│   ├── TROUBLESHOOTING.md   # Common issues & solutions
│   └── *.md                 # Additional documentation
├── 📂 scripts/               # Setup & utility scripts
│   ├── setup.sh            # One-command setup script
│   ├── setup.py            # Python environment setup
│   └── cleanup_project.sh  # Project cleanup utilities
├── 📂 tests/                 # Test files
│   └── test_prediction.py  # Model validation tests
├── 📂 config/                # Configuration files
│   └── requirements.txt    # Python dependencies
└── 📂 backup/               # Backup of removed files
    └── backup_removed_files_*/  # Timestamped cleanup backups
```
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - RandomForest meta-learner
    - Weighted ensemble approach
```

## 📊 Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features
- **Text**: Combined review content per listing

### Feature Engineering
1. **Property Features**
   - Space efficiency ratios
   - Amenity categorization
   - Location clustering
   - Host experience metrics

2. **Review Processing**
   - Text aggregation by listing
   - Sentiment and content analysis via DistilBERT
   - Review count and velocity features

3. **Advanced Preprocessing**
   - Power transformations for skewed features
   - Quantile normalization
   - One-hot encoding for categories
   - Log transformation for prices

## 🔧 Technical Implementation

### Key Classes

#### 1. DistilBertTextEncoder
```python
class DistilBertTextEncoder(BaseEstimator, RegressorMixin):
    def __init__(self, max_length=128, batch_size=16):
        # Initialize DistilBERT for text encoding
    
    def fit(self, X, y=None):
        # Load pre-trained DistilBERT model
    
    def transform(self, X):
        # Convert text to 768-dim embeddings
```

#### 2. MultimodalRegressor
```python
class MultimodalRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, tabular_model, text_encoder, meta_model):
        # Combine tabular and text models
    
    def fit(self, X_tabular, X_text, y):
        # Train both modalities + meta-learner
    
    def predict(self, X_tabular, X_text):
        # Generate multimodal predictions
```

## 🖥️ User Interface Features

### **Modern Design**
- 🎨 **Glassmorphism UI**: Beautiful, modern interface with transparency effects
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🌈 **Custom Styling**: Hand-crafted CSS with Airbnb brand colors
- ⚡ **Fast Performance**: Optimized rendering and caching

### **Interactive Prediction**
- 🏠 **Property Input Form**: Comprehensive sidebar form for all property details
- 📊 **Real-time Predictions**: Instant price predictions as you type
- 🔄 **Session Persistence**: Maintains predictions when exploring different analyses
- 💾 **State Management**: Smart caching prevents data loss during interaction

### **Explainable AI Dashboard**
- 📈 **Feature Importance Charts**: Interactive bar charts and pie charts
- 🎯 **Top 5 Features**: Beautiful metric cards showing most important factors
- 📊 **Complete Rankings**: Sortable tables with all feature importance scores
- 🔍 **Feature Categories**: Organized analysis by property types, amenities, etc.

### **Sensitivity Analysis**
- 📉 **Interactive Charts**: See how changing features affects price
- 🎛️ **Dynamic Controls**: Real-time updates as you select different features
- 📍 **Current Value Markers**: Clear indication of your property's current position
- � **Price Curves**: Smooth visualization of price sensitivity

### **Smart Recommendations**
- 💡 **Actionable Insights**: Specific suggestions for price optimization
- ⚖️ **Strength Analysis**: What's working well for your pricing
- ⚠️ **Improvement Areas**: Features that might be reducing your price
- 📈 **Opportunity Identification**: Potential for price increases

---

## 🏗️ Architecture

### 1. Multimodal Design
```
┌─────────────────┐    ┌─────────────────┐
│   Tabular Data  │    │   Review Text   │
│  (Properties)   │    │  (NLP Features) │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│ Ensemble Model  │    │ DistilBERT      │
│ (RF+GB+ET)     │    │ Text Encoder    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Meta-Learner   │
            │ (RandomForest)  │
            └─────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Final Prediction│
            └─────────────────┘
```

### 2. Data Modalities

#### Tabular Features (85.7% R² baseline)
- **Property characteristics**: bedrooms, bathrooms, accommodates
- **Location features**: neighborhood, distance from center
- **Host information**: superhost status, experience, listings count
- **Amenities**: 20+ categorized amenities (luxury, tech, convenience)
- **Pricing ratios**: price per person, space efficiency
- **Availability**: booking flexibility, availability rate
- **Review metrics**: count, velocity, quality scores

#### Text Features (NLP Enhancement)
- **Review aggregation**: Combined guest reviews per listing
- **DistilBERT encoding**: 768-dimensional semantic embeddings
- **Preprocessing**: Text cleaning, normalization, tokenization
- **Batch processing**: Efficient GPU/CPU inference

### 3. Model Components

#### Tabular Ensemble (3 Models)
```python
models = {
    'ExtraTreesUltra': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, 
        min_samples_split=2, min_samples_leaf=1
    ),
    'GradientBoostingUltra': GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7
    ),
    'RandomForestUltra': RandomForestRegressor(
        n_estimators=500, max_depth=30, 
        min_samples_split=5, min_samples_leaf=2
    )
}
```

#### Text Processing Pipeline
```python
class DistilBertTextEncoder:
    - Tokenization with DistilBERT tokenizer
    - 768-dimensional embeddings
    - Batch processing for efficiency
    - GPU acceleration when available
```

#### Meta-Learning Fusion
```python
class MultimodalRegressor:
    - Combines tabular predictions + text embeddings
    - Random