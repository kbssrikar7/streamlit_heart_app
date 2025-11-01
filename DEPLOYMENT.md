# Deployment Guide for Streamlit Cloud

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name it (e.g., `heart-attack-risk-predictor`)
4. Choose **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Connect Your Local Repository to GitHub

After creating the GitHub repository, you'll see instructions. Run these commands in your terminal:

```bash
cd /Users/happy/Documents/Code/streamlit_heart_app

# Add your GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push your code to GitHub
git push -u origin main
```

**Or if you're using SSH:**
```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

## Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in the details:
   - **Repository**: Select your newly created repository
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a unique name (e.g., `heart-attack-risk-predictor`)
5. Click "Deploy!"

## Step 4: Wait for Deployment

Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Load your models from the `models/` directory
- Start your app

Deployment usually takes 1-2 minutes.

## Troubleshooting

### If deployment fails:

1. **Check logs**: Click on your app → "Manage app" → "Logs"
2. **Common issues**:
   - Missing dependencies: Check `requirements.txt`
   - Model files too large: GitHub has a 100MB file limit
   - Import errors: Check that all Python files are correct

### File Size Issues:

If your model files exceed GitHub's 100MB limit, consider:
- Using Git LFS (Large File Storage)
- Storing models on cloud storage (S3, Google Drive) and loading them at runtime
- Compressing models before committing

### Quick Git Commands Reference:

```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Your message"

# Push to GitHub
git push origin main
```

## Your Repository is Ready!

All files are committed and ready to push. Just connect it to GitHub and deploy!



