#!/usr/bin/env python3
"""
🔍 Project File Analyzer
========================

Quick script to analyze and categorize all files in the Airbnb Price Predictor project.
Helps identify which files are essential, legacy, or can be removed.
"""

import os
import glob
from pathlib import Path
from datetime import datetime

def get_file_size(filepath):
    """Get file size in human readable format"""
    try:
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "Unknown"

def analyze_files():
    """Analyze all files in the current directory"""
    
    # File categories
    categories = {
        'ESSENTIAL': {
            'files': ['streamlit_app.py', 'code.ipynb', 'listings.csv', 'reviews.csv', 
                     'requirements.txt', 'README.md'],
            'description': 'Core application files - DO NOT REMOVE'
        },
        'CURRENT_MODELS': {
            'files': ['model_data_for_streamlit.json', 'model_state.json', 'training_data_export.json'],
            'description': 'Current production models - KEEP'
        },
        'SETUP_DOCS': {
            'files': ['setup.sh', 'setup.py', 'TROUBLESHOOTING.md', 'PROJECT_STRUCTURE.md',
                     'NUMPY_FIX_SUMMARY.md'],
            'description': 'Setup and documentation - USEFUL'
        },
        'USEFUL_TOOLS': {
            'files': ['demo.py', 'test_prediction.py', 'preprocessor_simple.pkl'],
            'description': 'Demo and testing tools - CONSIDER KEEPING'
        },
        'LEGACY_PICKLE': {
            'files': ['multimodal_airbnb_model.pkl', 'multimodal_airbnb_model_v2.pkl',
                     'multimodal_model_clean.pkl', 'tabular_airbnb_model.pkl',
                     'tabular_airbnb_model_v2.pkl', 'tabular_model_clean.pkl',
                     'preprocessor.pkl', 'preprocessor_clean.pkl', 'preprocessor_v2.pkl',
                     'metadata.pkl', 'metadata_clean.pkl', 'metadata_v2.pkl'],
            'description': 'Legacy pickle models - SAFE TO REMOVE'
        },
        'LEGACY_JSON': {
            'files': ['streamlit_complete_model.json', 'streamlit_linear_model.json',
                     'streamlit_simple_model.json'],
            'description': 'Legacy JSON models - SAFE TO REMOVE'
        },
        'DEBUG_TEST': {
            'files': ['debug_shape_error.py', 'test_dataframe.py', 'test_dataframe_fix.py',
                     'test_explanation.py', 'test_final_fix.py', 'test_shape_fix.py',
                     'test_clean_models.py', 'streamlit_app_json.py'],
            'description': 'Debug and test files - SAFE TO REMOVE'
        }
    }
    
    print("🔍 AIRBNB PRICE PREDICTOR - FILE ANALYSIS")
    print("=" * 50)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    total_files = 0
    total_size = 0
    
    for category, info in categories.items():
        print(f"📁 {category}")
        print(f"   {info['description']}")
        print("   " + "-" * 40)
        
        category_files = 0
        category_size = 0
        
        for filename in info['files']:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                size_str = get_file_size(filename)
                status = "✅ EXISTS"
                category_files += 1
                category_size += size
            else:
                size_str = "N/A"
                status = "❌ MISSING"
            
            print(f"   {status:<12} {filename:<35} {size_str:>10}")
        
        print(f"   📊 Category Total: {category_files} files, {get_file_size(category_size)}")
        print()
        
        total_files += category_files
        total_size += category_size
    
    # Check for unclassified files
    all_classified = []
    for category in categories.values():
        all_classified.extend(category['files'])
    
    unclassified = []
    for file in glob.glob("*"):
        if os.path.isfile(file) and file not in all_classified:
            # Skip system files
            if not file.startswith('.') and file not in ['cleanup_project.sh', 'analyze_files.py']:
                unclassified.append(file)
    
    if unclassified:
        print("🔍 UNCLASSIFIED FILES")
        print("   Files not in the above categories")
        print("   " + "-" * 40)
        for filename in unclassified:
            size_str = get_file_size(filename)
            print(f"   ⚠️  UNKNOWN   {filename:<35} {size_str:>10}")
        print()
    
    # Check directories
    directories = [d for d in glob.glob("*") if os.path.isdir(d) and not d.startswith('.')]
    if directories:
        print("📂 DIRECTORIES")
        print("   " + "-" * 40)
        for dirname in directories:
            if dirname == '__pycache__':
                print(f"   🗑️  CACHE     {dirname:<35} (auto-generated)")
            elif dirname == '.venv':
                print(f"   ✅ ESSENTIAL {dirname:<35} (virtual environment)")
            elif dirname == 'model_artifacts':
                print(f"   ⚠️  LEGACY    {dirname:<35} (backup models)")
            else:
                print(f"   📁 DIRECTORY {dirname:<35}")
        print()
    
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total Classified Files: {total_files}")
    print(f"Total Size: {get_file_size(total_size)}")
    print()
    
    # Recommendations
    removable_categories = ['LEGACY_PICKLE', 'LEGACY_JSON', 'DEBUG_TEST']
    removable_files = 0
    removable_size = 0
    
    for category in removable_categories:
        if category in categories:
            for filename in categories[category]['files']:
                if os.path.exists(filename):
                    removable_files += 1
                    removable_size += os.path.getsize(filename)
    
    print("💡 CLEANUP RECOMMENDATIONS")
    print("=" * 50)
    print(f"🗑️  Files safe to remove: {removable_files}")
    print(f"💾 Space that can be freed: {get_file_size(removable_size)}")
    print()
    print("🚀 To clean up the project:")
    print("   ./cleanup_project.sh")
    print()
    print("📚 For detailed file documentation:")
    print("   cat PROJECT_STRUCTURE.md")

if __name__ == "__main__":
    analyze_files()
