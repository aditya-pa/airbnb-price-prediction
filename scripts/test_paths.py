#!/usr/bin/env python3
"""
Test script to verify that file paths are working correctly after reorganization
"""

import os
import sys

def test_file_paths():
    """Test if all expected files exist in their new locations"""
    
    # Get project root directory
    if 'scripts' in os.getcwd():
        # Running from scripts directory
        project_root = os.path.dirname(os.getcwd())
    else:
        # Running from project root or other directory
        project_root = os.getcwd()
    
    print(f"🔍 Testing file paths from: {project_root}")
    print("=" * 50)
    
    # Expected file structure
    expected_files = {
        'src/streamlit_app.py': 'Main Streamlit application',
        'src/demo.py': 'Demo script',
        'notebooks/code.ipynb': 'ML training notebook',
        'data/listings.csv': 'Property data',
        'data/reviews.csv': 'Reviews data',
        'models/model_data_for_streamlit.json': 'Model data',
        'models/model_state.json': 'Model state',
        'models/preprocessor_simple.pkl': 'Preprocessor',
        'config/requirements.txt': 'Python dependencies',
        'docs/README.md': 'Main documentation',
        'tests/test_prediction.py': 'Test script',
        'scripts/setup.sh': 'Setup script'
    }
    
    # Test each file
    all_good = True
    for file_path, description in expected_files.items():
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path:<35} - {description}")
        else:
            print(f"❌ {file_path:<35} - {description} (MISSING)")
            all_good = False
    
    print("=" * 50)
    
    if all_good:
        print("🎉 All files found in correct locations!")
        print("✅ File organization is working correctly")
    else:
        print("⚠️  Some files are missing from expected locations")
        print("💡 Check if files were moved correctly")
    
    # Test Python imports from src directory
    print("\n🐍 Testing Python path resolution...")
    try:
        sys.path.insert(0, os.path.join(project_root, 'src'))
        
        # Test if we can determine correct paths
        test_script_dir = os.path.join(project_root, 'src')
        base_dir = os.path.dirname(test_script_dir)  # Should be project_root
        models_dir = os.path.join(base_dir, 'models')
        
        if os.path.exists(models_dir):
            print(f"✅ Models directory path resolved: {models_dir}")
        else:
            print(f"❌ Models directory not found: {models_dir}")
            
    except Exception as e:
        print(f"❌ Python path resolution failed: {e}")
    
    return all_good

if __name__ == "__main__":
    print("🧪 File Path Organization Test")
    print("Testing if all files are in their correct locations after reorganization...")
    print()
    
    success = test_file_paths()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Run: streamlit run src/streamlit_app.py")
        print("2. Or run: python tests/test_prediction.py")
        print("3. For setup: ./scripts/setup.sh")
        exit(0)
    else:
        print("\n❌ File organization needs attention")
        exit(1)
