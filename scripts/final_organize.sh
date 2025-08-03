#!/bin/bash
# Final cleanup of docs directory and overall organization

echo "🧹 Final repository organization..."

cd "/Users/adityapandey/My Files/Thesis Sri Ganesh/Data Set/7"

# Clean up docs directory
echo "📁 Organizing docs directory..."

# Keep only essential documentation files in docs
rm -f docs/CHANGELOG.md
rm -f docs/PROJECT_STRUCTURE.md  
rm -f docs/PROJECT_STRUCTURE_NEW.md
rm -f docs/README_NEW.md
rm -rf docs/archive

# Move images to a proper images directory
if [ ! -d "docs/images" ]; then
    mkdir -p docs/images
fi

mv docs/*.png docs/images/ 2>/dev/null || true
echo "   ✅ Moved image files to docs/images/"

# Keep only essential files in docs root
echo "📋 Essential docs files:"
ls docs/

# Clean up any remaining unnecessary directories
echo ""
echo "🧹 Final cleanup..."

# Remove backup and tests if they're empty
rmdir backup 2>/dev/null && echo "   ✅ Removed empty backup directory" || echo "   ℹ️  Backup directory contains files (kept)"
rmdir tests 2>/dev/null && echo "   ✅ Removed empty tests directory" || echo "   ℹ️  Tests directory contains files (kept)"

echo ""
echo "✨ Final organization complete!"
echo ""
echo "📊 Repository structure summary:"
echo "📁 Root files: $(find . -maxdepth 1 -type f | wc -l) files"
echo "📁 Source code: $(find src -name '*.py' 2>/dev/null | wc -l) Python files"  
echo "📁 Models: $(find models -name '*.pkl' -o -name '*.json' 2>/dev/null | wc -l) model files"
echo "📁 Data files: $(find data -type f 2>/dev/null | wc -l) data files"
echo "📁 Notebooks: $(find notebooks -name '*.ipynb' 2>/dev/null | wc -l) notebooks"
echo "📁 Documentation: $(find docs -name '*.md' 2>/dev/null | wc -l) markdown files"
