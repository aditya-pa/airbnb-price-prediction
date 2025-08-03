#!/bin/bash
# Quick Repository Cleanup Script
# Removes unnecessary files and organizes the repository

echo "🧹 Starting repository cleanup..."

# Remove .DS_Store files
echo "🗑️  Removing .DS_Store files..."
find . -name ".DS_Store" -type f -delete

# Remove duplicate model files from notebooks directory
echo "🗑️  Removing duplicate model files from notebooks..."
rm -f notebooks/*.pkl
rm -f notebooks/*.json
rm -rf notebooks/model_artifacts

# Remove duplicate markdown documentation files
echo "🗑️  Removing duplicate markdown files..."
rm -f DATA_COMPREHENSIVE_ANALYSIS_REPORT.md
rm -f DETAILED_TECHNICAL_PROCESS.md
rm -f PROJECT_LESSONS_LEARNED.md  
rm -f PROJECT_ORGANIZATION_COMPLETE.md
rm -f STATISTICAL_FINDINGS_SUMMARY.md
rm -f THESIS_EXECUTIVE_SUMMARY.md

# Create docs structure if it doesn't exist
echo "📁 Creating organized directory structure..."
mkdir -p docs/deployment
mkdir -p docs/reports

# Move deployment files to docs/deployment
echo "📁 Moving deployment files..."
if [ -f "DEPLOYMENT.md" ]; then
    mv DEPLOYMENT.md docs/deployment/
    echo "   ✅ Moved DEPLOYMENT.md"
fi

if [ -f "DEPLOY_CHECKLIST.md" ]; then
    mv DEPLOY_CHECKLIST.md docs/deployment/
    echo "   ✅ Moved DEPLOY_CHECKLIST.md"
fi

# Remove empty directories
echo "🧹 Cleaning up empty directories..."
rmdir backup 2>/dev/null && echo "   ✅ Removed empty backup directory" || true
rmdir tests 2>/dev/null && echo "   ✅ Removed empty tests directory" || true

echo ""
echo "✨ Cleanup complete! Repository is now organized."
echo ""
echo "📋 Current structure:"
echo "├── 📁 src/           # Source code"
echo "├── 📁 models/        # ML model files"  
echo "├── 📁 data/          # Dataset files"
echo "├── 📁 notebooks/     # Jupyter notebooks"
echo "├── 📁 scripts/       # Utility scripts"
echo "├── 📁 docs/          # Documentation"
echo "│   ├── 📁 deployment/    # Deployment guides"
echo "│   └── 📁 reports/       # Analysis reports"
echo "├── 📄 README.md      # Main documentation"
echo "├── 📄 requirements.txt   # Dependencies"
echo "└── 📄 app.py         # Main app entry point"
