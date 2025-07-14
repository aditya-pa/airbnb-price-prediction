# Streamlit App Configuration for Deployment
# This file ensures the app uses lightweight model loading for deployment

import os
import warnings

# Deployment settings
DEPLOYMENT_MODE = os.environ.get('STREAMLIT_SHARING_MODE', 'False').lower() == 'true' or \
                 os.environ.get('RAILWAY_ENVIRONMENT', None) is not None or \
                 os.environ.get('HUGGINGFACE_SPACES', None) is not None

# Suppress warnings in deployment
if DEPLOYMENT_MODE:
    warnings.filterwarnings('ignore')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Model loading preferences
PREFER_LIGHTWEIGHT_MODELS = True
MAX_MODEL_SIZE_MB = 100

print(f"Deployment mode: {DEPLOYMENT_MODE}")
print(f"Using lightweight models: {PREFER_LIGHTWEIGHT_MODELS}")
