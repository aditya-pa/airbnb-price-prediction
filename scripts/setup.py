import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a command and return True if successful"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🏠 Setting up Airbnb Price Predictor with Explainable AI...")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} found")
    
    # Check if pip is available
    if not run_command("pip --version"):
        print("❌ pip is not installed or not in PATH")
        return False
    
    print("✅ pip found")
    
    # Ask about virtual environment
    create_venv = input("Do you want to create a virtual environment? (y/n): ").lower().strip()
    
    if create_venv in ['y', 'yes']:
        print("📦 Creating virtual environment...")
        
        if not run_command("python -m venv airbnb_predictor_env"):
            print("❌ Failed to create virtual environment")
            return False
        
        print("✅ Virtual environment created")
        print("💡 To activate in the future:")
        if platform.system() == "Windows":
            print("   airbnb_predictor_env\\Scripts\\activate")
        else:
            print("   source airbnb_predictor_env/bin/activate")
    
    # Install requirements
    print("📥 Installing required packages...")
    
    if not run_command("pip install -r requirements.txt"):
        print("❌ Error installing packages")
        return False
    
    print("✅ All packages installed successfully")
    
    # Check for data files
    if not os.path.exists("listings.csv"):
        print("⚠️  Warning: listings.csv not found")
        print("   Please ensure you have the Airbnb listings data file")
    
    if not os.path.exists("reviews.csv"):
        print("⚠️  Warning: reviews.csv not found")
        print("   Please ensure you have the Airbnb reviews data file")
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Ensure you have listings.csv and reviews.csv in this directory")
    print("2. Run the training notebook: jupyter notebook code.ipynb")
    print("3. Launch the web app: streamlit run streamlit_app.py")
    print()
    print("📚 Check README.md for detailed instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
