#!/bin/bash

# Airbnb Price Predictor Setup Script

echo "🏠 Setting up Airbnb Price Predictor with Explainable AI..."
echo "=================================================="

# Get the project root directory (parent of scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n): " create_venv

if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv airbnb_predictor_env
    
    echo "🔧 Activating virtual environment..."
    source airbnb_predictor_env/bin/activate
    
    echo "✅ Virtual environment created and activated"
    echo "💡 To activate in the future, run: source airbnb_predictor_env/bin/activate"
fi

# Install requirements
echo "📥 Installing required packages..."
pip install -r config/requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "❌ Error installing packages. Please check the error messages above."
    exit 1
fi

# Check if data files exist
if [ ! -f "../data/listings.csv" ]; then
    echo "⚠️  Warning: listings.csv not found in data/ folder"
    echo "   Please ensure you have the Airbnb listings data file"
fi

if [ ! -f "../data/reviews.csv" ]; then
    echo "⚠️  Warning: reviews.csv not found in data/ folder"
    echo "   Please ensure you have the Airbnb reviews data file"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Ensure you have listings.csv and reviews.csv in the data/ directory"
echo "2. Run the training notebook: jupyter notebook notebooks/code.ipynb"
echo "3. Launch the web app: streamlit run src/streamlit_app.py"
echo ""
echo "📚 Check docs/README.md for detailed instructions"
