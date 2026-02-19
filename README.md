# Airbnb Smart Pricing Engine

A comprehensive machine learning solution that predicts Airbnb prices using both tabular property data and guest review text, with built-in explainability features and an interactive web interface for property owners.

## Key Features

- **Multimodal AI**: Combines property features with guest review sentiment using DistilBERT
- **Explainable AI**: SHAP-based explanations showing which features impact pricing
- **Interactive UI**: Beautiful Streamlit web app with modern design
- **Sensitivity Analysis**: Interactive charts showing how features affect price
- **Actionable Insights**: Get specific recommendations for price optimization
- **Feature Importance**: Comprehensive analysis of what drives pricing decisions
- **Responsive Design**: Mobile-friendly interface with glassmorphism styling

## Model Performance

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

## Quick Start

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

## Project Structure

```
Airbnb Smart Pricing Engine
├── src/                    # Main application code
│   ├── streamlit_app.py       # Primary Streamlit web app
│   └── demo.py               # Model demonstration script
├── notebooks/             # Jupyter notebooks
│   └── code.ipynb           # Complete ML pipeline & training
├── data/                  # Dataset files
│   ├── listings.csv         # Property features (primary dataset)
│   └── reviews.csv          # Guest reviews for sentiment analysis
├── models/                # Model artifacts & data
│   ├── model_data_for_streamlit.json  # Production model data
│   ├── model_state.json               # Model state backup
│   ├── training_data_export.json      # Training data export
│   └── preprocessor_simple.pkl        # Data preprocessor
├── docs/                  # Documentation
│   ├── README.md            # This file
│   ├── PROJECT_STRUCTURE.md # Detailed file organization
│   ├── TROUBLESHOOTING.md   # Common issues & solutions
│   └── *.md                 # Additional documentation
├── scripts/               # Setup & utility scripts
│   ├── setup.sh            # One-command setup script
│   ├── setup.py            # Python environment setup
│   └── cleanup_project.sh  # Project cleanup utilities
├── tests/                 # Test files
│   └── test_prediction.py  # Model validation tests
├── config/                # Configuration files
│   └── requirements.txt    # Python dependencies
└── backup/               # Backup of removed files
    └── backup_removed_files_*/  # Timestamped cleanup backups
```


---

## Architecture

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
│ (RF+GB+ET)      │    │ Text Encoder    │
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

## Data Processing

### Input Data
- **Listings**: 6,481 properties
- **Reviews**: 293,744 guest reviews
- **Features**: 30+ engineered features

### Feature Engineering
- **Categorical encoding**: One-hot encoding for neighborhood, property type
- **Numerical scaling**: StandardScaler for continuous features
- **Text processing**: DistilBERT embeddings for review sentiment
- **Feature creation**: Ratios, interactions, polynomial features

### Data Pipeline
```python
# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])
```

## 📖 Usage Guide

### **Step-by-Step Usage**

#### 1. **Launch the Application**
```bash
cd "/path/to/project"
streamlit run src/streamlit_app.py
```

#### 2. **Input Property Details**
- **Basic Info**: Set bedrooms, bathrooms, guest capacity
- **Pricing**: Enter current price and availability settings
- **Amenities**: Check available amenities (WiFi, kitchen, parking, etc.)
- **Location**: Select neighborhood and property type
- **Reviews**: Add sample guest review text (optional but recommended)

#### 3. **Generate Predictions**
- Click **"Predict Price & Explain"** 
- View three price predictions:
  - **Current Price**: Your input price
  - **Tabular Model**: Price based on property features only
  - **Full Model**: Price including review sentiment analysis

#### 4. **Explore Explanations**
- **Model Explanation**: See SHAP values showing feature impact
- **Feature Importance**: Interactive charts of what drives pricing
- **Sensitivity Analysis**: Test how changing features affects price
- **Recommendations**: Get actionable pricing optimization tips

### **Understanding the Results**

#### **Price Predictions**
- **Green percentages**: Model suggests you can increase price
- **Red percentages**: Model suggests price might be too high
- **Difference between models**: Shows value of including review sentiment

#### **Feature Importance**
- **Top 5 Cards**: Most critical features for your property type
- **Bar Charts**: Comprehensive ranking of all features
- **Pie Chart**: Feature categories (Property, Location, Amenities, etc.)
- **Complete Table**: Detailed scores and rankings

#### **Sensitivity Analysis**
- **Interactive Dropdown**: Select different features to analyze
- **Price Curve**: Shows how price changes with feature values
- **Current Value Line**: Your property's position on the curve
- **Optimization Insights**: Identify optimal feature values

## Technical Details

### **Performance Metrics**
- **R² Score**: 86.0% (vs 85.1% tabular-only)
- **MAE**: $26.72 (vs $28.45 tabular-only)
- **Cross-Validation**: 85.1% ± 2.0% (excellent stability)
- **Improvement**: +0.9% R², +6.1% MAE from multimodal approach

### **Dependencies**
```python
# Core ML
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3

# NLP
transformers==4.33.0
torch==2.0.1

# Web Interface
streamlit==1.28.0
plotly==5.15.0

# Explainability
shap==0.42.1
```

### **Performance Optimizations**
- **Batch Processing**: DistilBERT processes reviews in batches
- **Model Caching**: Streamlit caches loaded models
- **Efficient Preprocessing**: Optimized feature engineering pipeline
- **Memory Management**: Careful handling of large embeddings

## Deployment

### **Local Development**
```bash
# Clone and setup
git clone <repository-url>
cd airbnb-pricing-engine
./scripts/setup.sh
streamlit run src/streamlit_app.py
```

### **Production Deployment**
```bash
# Using Docker (example)
docker build -t airbnb-pricing .
docker run -p 8501:8501 airbnb-pricing

# Using cloud platforms
# - Streamlit Cloud
# - Heroku
# - AWS/GCP/Azure
```

## Live Demo & Deployment

### **For Teachers & Recruiters**
Access the live application: [Coming Soon - Deploy Instructions Below]

### **Deploy Your Own (FREE)**

**Option 1: Streamlit Community Cloud (Recommended)**
1. Visit: https://streamlit.io/cloud
2. Connect your GitHub account
3. Deploy this repository with main file: `src/streamlit_app.py`

**Option 2: Hugging Face Spaces**
1. Visit: https://huggingface.co/spaces
2. Create new Streamlit space
3. Upload files or connect GitHub

**Option 3: Railway**
1. Visit: https://railway.app
2. Connect GitHub repository
3. Auto-deploy detected

📋 **See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions**

---
## Use Cases

### For Property Owners
- **Price Optimization**: Set competitive prices based on market data
- **Feature Planning**: Understand which amenities add most value
- **Market Analysis**: Compare your property against similar listings
- **Investment Decisions**: Evaluate property improvement ROI

### For Data Scientists
- **Multimodal Learning**: Example of combining tabular and text data
- **Explainable AI**: Implementation of SHAP for model interpretation
- **Ensemble Methods**: Advanced ensemble techniques for regression
- **Production Deployment**: End-to-end ML system with web interface

## Customization

### Adding New Features
1. Update feature engineering in `notebooks/code.ipynb`
2. Retrain models with new features
3. Update UI form in `src/streamlit_app.py`

### Model Improvements
- **Text Processing**: Try different pre-trained models (BERT, RoBERTa)
- **Ensemble Methods**: Add more diverse base models
- **Feature Engineering**: Create domain-specific features

## Support

- **Documentation**: Check `docs/` folder for detailed guides
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`
- **Project Structure**: Refer to `docs/PROJECT_STRUCTURE.md`
- **Issues**: Create GitHub issues for bugs or feature requests

## Acknowledgments

- **Scikit-learn**: For excellent machine learning tools
- **Hugging Face**: For DistilBERT and transformers library
- **Streamlit**: For the beautiful web framework
- **SHAP**: For explainable AI capabilities
- **Plotly**: For interactive visualizations

---

**Built with for the Airbnb host community**

*Empowering property owners with AI-driven pricing intelligence*

---

*Last Updated: July 7, 2025*  
*Version: 2.0 - Clean & Organized* ✨
