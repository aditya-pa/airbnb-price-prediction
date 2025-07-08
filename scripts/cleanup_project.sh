#!/bin/bash

# 🧹 Airbnb Price Predictor - Smart Project Cleanup Script
# This script removes unnecessary legacy files while preserving all essential functionality

echo "🏠 Airbnb Smart Pricing Engine - Project Cleanup"
echo "=================================================="
echo "🧹 Starting intelligent cleanup process..."
echo ""

# Get current directory size
echo "📊 Current project size:"
du -sh . 2>/dev/null | head -1
echo ""

# Create timestamped backup directory for safety
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_removed_files_${TIMESTAMP}"
echo "� Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"/{debug_scripts,legacy_models,legacy_json,misc}

# Function to safely remove files with backup
safe_remove() {
    local file="$1"
    local category="$2"
    if [ -f "$file" ]; then
        echo "  📁 Backing up: $file"
        cp "$file" "$BACKUP_DIR/$category/" 2>/dev/null
        echo "  🗑️  Removing: $file" 
        rm "$file"
        return 0
    else
        return 1
    fi
}

# 🔧 PHASE 1: Remove debug and test scripts (no longer needed)
echo "🔧 Phase 1: Removing debug and test scripts..."
debug_files=(
    "debug_shape_error.py"
    "test_dataframe.py"
    "test_dataframe_fix.py"
    "test_explanation.py"
    "test_final_fix.py"
    "test_shape_fix.py"
    "test_clean_models.py"
    "streamlit_app_json.py"
    "analyze_files.py"
)

debug_removed=0
for file in "${debug_files[@]}"; do
    if safe_remove "$file" "debug_scripts"; then
        ((debug_removed++))
    fi
done
echo "   ✅ Removed $debug_removed debug scripts"
echo ""

# 🤖 PHASE 2: Remove legacy pickle models (superseded by JSON)
echo "🤖 Phase 2: Removing legacy pickle model files..."
model_files=(
    "multimodal_airbnb_model.pkl"
    "multimodal_airbnb_model_v2.pkl"
    "multimodal_model_clean.pkl"
    "tabular_airbnb_model.pkl"
    "tabular_airbnb_model_v2.pkl"
    "tabular_model_clean.pkl"
    "preprocessor.pkl"
    "preprocessor_clean.pkl"
    "preprocessor_v2.pkl"
    "metadata.pkl"
    "metadata_clean.pkl"
    "metadata_v2.pkl"
)

models_removed=0
for file in "${model_files[@]}"; do
    if safe_remove "$file" "legacy_models"; then
        ((models_removed++))
    fi
done
echo "   ✅ Removed $models_removed legacy pickle models"
echo ""

# 📄 PHASE 3: Remove legacy JSON models (superseded)
echo "📄 Phase 3: Removing legacy JSON model files..."
json_files=(
    "streamlit_complete_model.json"
    "streamlit_linear_model.json"
    "streamlit_simple_model.json"
)

json_removed=0
for file in "${json_files[@]}"; do
    if safe_remove "$file" "legacy_json"; then
        ((json_removed++))
    fi
done
echo "   ✅ Removed $json_removed legacy JSON models"
echo ""

# 🧽 PHASE 4: Clean Python cache and temporary files
echo "🧽 Phase 4: Cleaning Python cache and temporary files..."
cache_cleaned=0

# Remove __pycache__ directories
if find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; then
    cache_cleaned=1
fi

# Remove .pyc files
pyc_count=$(find . -name "*.pyc" | wc -l)
find . -name "*.pyc" -delete 2>/dev/null

# Remove .pyo files  
pyo_count=$(find . -name "*.pyo" | wc -l)
find . -name "*.pyo" -delete 2>/dev/null

echo "   ✅ Cleaned Python cache ($pyc_count .pyc files, $pyo_count .pyo files)"
echo ""

# 🗂️ PHASE 5: Handle model_artifacts directory (optional)
echo "🗂️  Phase 5: Checking model_artifacts directory..."
if [ -d "model_artifacts" ]; then
    artifacts_size=$(du -sh model_artifacts 2>/dev/null | cut -f1)
    echo "   📁 Found model_artifacts directory ($artifacts_size)"
    echo "   � Backing up model_artifacts..."
    cp -r model_artifacts "$BACKUP_DIR/"
    echo "   ⚠️  Keeping model_artifacts (you can remove manually if JSON models work perfectly)"
    echo "   💡 To remove: rm -rf model_artifacts"
else
    echo "   ✅ No model_artifacts directory found"
fi
echo ""

# ✅ PHASE 6: Verify essential files are still present
echo "✅ Phase 6: Verifying essential files are present..."
essential_files=(
    "streamlit_app.py"
    "code.ipynb"
    "listings.csv"
    "reviews.csv"
    "model_data_for_streamlit.json"
    "requirements.txt"
    "README.md"
    "TROUBLESHOOTING.md"
)

missing_files=()
present_files=0
for file in "${essential_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
        ((present_files++))
    else
        echo "   ❌ $file - MISSING!"
        missing_files+=("$file")
    fi
done
echo "   📊 Essential files present: $present_files/${#essential_files[@]}"
echo ""

# 📊 PHASE 7: Show cleanup results and statistics
echo "🎉 CLEANUP COMPLETED!"
echo "===================="
echo ""
echo "📊 Cleanup Statistics:"
echo "   🔧 Debug scripts removed: $debug_removed"
echo "   🤖 Legacy models removed: $models_removed" 
echo "   📄 Legacy JSON removed: $json_removed"
echo "   🧽 Python cache cleaned: Yes"
echo "   📁 Total files removed: $((debug_removed + models_removed + json_removed))"
echo ""

echo "💾 Backup Information:"
echo "   📂 Backup location: ./$BACKUP_DIR/"
echo "   🗂️  Backup contents:"
echo "      • debug_scripts/ - $debug_removed files"
echo "      • legacy_models/ - $models_removed files" 
echo "      • legacy_json/ - $json_removed files"
if [ -d "model_artifacts" ]; then
    echo "      • model_artifacts/ - backup copy"
fi
echo ""

echo "📊 Project size after cleanup:"
du -sh . 2>/dev/null | head -1
echo ""

# Show status
if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ SUCCESS: All essential files are present!"
    echo ""
    echo "🚀 Your Airbnb Smart Pricing Engine is ready!"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Run the application: streamlit run streamlit_app.py"
    echo "   2. Open browser to: http://localhost:8501"
    echo "   3. Start predicting prices!"
    echo ""
    echo "📖 Documentation:"
    echo "   • Main guide: README.md"
    echo "   • Complete file dictionary: FILE_DICTIONARY_COMPLETE.md"
    echo "   • Troubleshooting: TROUBLESHOOTING.md"
    echo ""
    echo "💡 If you encounter issues, restore from backup:"
    echo "   cp $BACKUP_DIR/[category]/[filename] ."
else
    echo "⚠️  WARNING: Missing essential files!"
    echo "Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   ❌ $file"
    done
    echo ""
    echo "🔧 Check your backup or restore from git."
fi

echo ""
echo "🧹 Cleanup script completed!"
echo "💡 Backup kept for safety: ./$BACKUP_DIR/"
