#!/bin/bash

# Script to push Transformer debugging assignment to GitHub
# Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME

if [ $# -eq 0 ]; then
    echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME"
    echo "Example: ./push_to_github.sh john-doe"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="transformer-debugging-assignment"
REPO_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo "🚀 Setting up GitHub repository for Transformer debugging assignment..."
echo "Repository URL: $REPO_URL"

# Check if we're in the right directory
if [ ! -f "transformer_model.py" ]; then
    echo "❌ Error: Please run this script from the Assignment directory"
    exit 1
fi

# Add remote origin
echo "📡 Adding remote origin..."
git remote add origin $REPO_URL

# Rename branch to main
echo "🔄 Renaming branch to main..."
git branch -M main

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Your repository is now available at: $REPO_URL"
    echo ""
    echo "📋 Next steps:"
    echo "1. Go to your repository on GitHub"
    echo "2. Add a description and topics"
    echo "3. Share the repository with your instructor"
    echo "4. Students can now clone and use the assignment"
else
    echo "❌ Error: Failed to push to GitHub"
    echo "Please check your GitHub username and repository URL"
    echo "Make sure you've created the repository on GitHub first"
fi
