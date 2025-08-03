#!/usr/bin/env python3
"""
Repository Cleanup and Organization Script
This script removes unnecessary files and organizes the repository structure properly.
"""

import os
import shutil
import sys
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧹 Starting repository cleanup and organization...")
    print(f"📂 Working directory: {project_root}")
    
    # Files and directories to remove
    files_to_remove = [
        # Duplicate markdown files
        "DATA_COMPREHENSIVE_ANALYSIS_REPORT.md",
        "DETAILED_TECHNICAL_PROCESS.md", 
        "PROJECT_LESSONS_LEARNED.md",
        "PROJECT_ORGANIZATION_COMPLETE.md",
        "STATISTICAL_FINDINGS_SUMMARY.md",
        "THESIS_EXECUTIVE_SUMMARY.md",
        
        # Duplicate model files in notebooks directory
        "notebooks/metadata.pkl",
        "notebooks/metadata_clean.pkl", 
        "notebooks/metadata_v2.pkl",
        "notebooks/model_data_for_streamlit.json",
        "notebooks/model_state.json",
        "notebooks/multimodal_airbnb_model.pkl",
        "notebooks/multimodal_airbnb_model_v2.pkl",
        "notebooks/multimodal_model_clean.pkl",
        "notebooks/preprocessor.pkl",
        "notebooks/preprocessor_clean.pkl",
        "notebooks/preprocessor_simple.pkl", 
        "notebooks/preprocessor_v2.pkl",
        "notebooks/streamlit_complete_model.json",
        "notebooks/streamlit_linear_model.json",
        "notebooks/streamlit_simple_model.json",
        "notebooks/tabular_airbnb_model.pkl",
        "notebooks/tabular_airbnb_model_v2.pkl",
        "notebooks/tabular_model_clean.pkl",
        "notebooks/model_artifacts",
        
        # System files
        ".DS_Store",
        "**/.DS_Store",
    ]
    
    # Directories to clean up
    dirs_to_clean = [
        "backup",  # Remove if empty or contains only old files
        "tests",   # Remove if empty
    ]
    
    print("\n🗑️  Removing unnecessary files...")
    removed_count = 0
    
    for file_pattern in files_to_remove:
        if "**/" in file_pattern:
            # Handle recursive patterns
            for file_path in Path(".").rglob(file_pattern.replace("**/", "")):
                if file_path.exists():
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            print(f"   ✅ Removed: {file_path}")
                            removed_count += 1
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            print(f"   ✅ Removed directory: {file_path}")
                            removed_count += 1
                    except Exception as e:
                        print(f"   ❌ Could not remove {file_path}: {e}")
        else:
            file_path = Path(file_pattern)
            if file_path.exists():
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"   ✅ Removed: {file_path}")
                        removed_count += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        print(f"   ✅ Removed directory: {file_path}")
                        removed_count += 1
                except Exception as e:
                    print(f"   ❌ Could not remove {file_path}: {e}")
    
    print(f"\n📊 Removed {removed_count} unnecessary files/directories")
    
    # Clean up empty directories
    print("\n🧹 Cleaning up empty directories...")
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            try:
                # Check if directory is empty or contains only .DS_Store files
                contents = list(dir_path.iterdir())
                if not contents or all(f.name == ".DS_Store" for f in contents):
                    shutil.rmtree(dir_path)
                    print(f"   ✅ Removed empty directory: {dir_path}")
                else:
                    print(f"   ℹ️  Kept non-empty directory: {dir_path}")
            except Exception as e:
                print(f"   ❌ Could not remove {dir_path}: {e}")
    
    # Organize files into proper structure
    print("\n📁 Organizing file structure...")
    
    # Create docs directory structure
    docs_structure = {
        "docs/deployment": ["DEPLOYMENT.md", "DEPLOY_CHECKLIST.md"],
        "docs": ["README.md"]  # Keep README at root but could move to docs if needed
    }
    
    # Ensure proper directory structure exists
    essential_dirs = [
        "src",
        "models", 
        "data",
        "notebooks",
        "scripts",
        "docs",
        "docs/deployment"
    ]
    
    for dir_name in essential_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created directory: {dir_path}")
    
    # Move deployment files to docs/deployment
    deployment_files = ["DEPLOYMENT.md", "DEPLOY_CHECKLIST.md"]
    for file_name in deployment_files:
        src_path = Path(file_name)
        dst_path = Path("docs/deployment") / file_name
        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"   ✅ Moved: {file_name} → docs/deployment/")
            except Exception as e:
                print(f"   ❌ Could not move {file_name}: {e}")
    
    print("\n✨ Repository cleanup and organization complete!")
    print("\n📋 Final structure:")
    print_directory_tree(".", max_depth=2)

def print_directory_tree(start_path, max_depth=2):
    """Print a simple directory tree"""
    start_path = Path(start_path)
    
    def _print_tree(path, prefix="", depth=0):
        if depth > max_depth:
            return
            
        if path.is_dir() and not path.name.startswith('.'):
            print(f"{prefix}📁 {path.name}/")
            if depth < max_depth:
                children = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
                for i, child in enumerate(children):
                    is_last = i == len(children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _print_tree(child, new_prefix, depth + 1)
        elif path.is_file() and not path.name.startswith('.'):
            icon = "📄" if path.suffix in ['.md', '.txt'] else "🐍" if path.suffix == '.py' else "📊" if path.suffix in ['.ipynb', '.json'] else "📄"
            print(f"{prefix}{icon} {path.name}")
    
    _print_tree(start_path)

if __name__ == "__main__":
    main()
