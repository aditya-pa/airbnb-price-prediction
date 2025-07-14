#!/usr/bin/env python3
"""
Airbnb Smart Pricing Engine
A machine learning app for predicting and explaining Airbnb prices

For deployment instructions and setup, see README.md
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
from streamlit_app import main

if __name__ == "__main__":
    main()
