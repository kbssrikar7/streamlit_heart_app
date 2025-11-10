# How to Clear Streamlit Cloud Cache

## Issue: App Showing Old Weights (20/80 instead of 50/50)

If your deployed Streamlit Cloud app is showing **20% XGBoost + 80% CatBoost** instead of **50% XGBoost + 50% CatBoost**, it's because Streamlit is caching the old model files.

## Solution: Clear Streamlit Cloud Cache

### Method 1: Clear Cache via Streamlit Cloud Dashboard (Recommended)

1. Go to your Streamlit Cloud app dashboard
2. Click on your app
3. Click **"⋮" (three dots)** menu → **"Clear cache"**
4. The app will restart with fresh cache

### Method 2: Redeploy the App

1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click **"⋮" (three dots)** menu → **"Reboot app"**
4. Wait for the app to restart

### Method 3: Force Cache Refresh (Code Change)

The code has been updated to use `@st.cache_resource(ttl=3600)` which will automatically refresh after 1 hour. But for immediate update:

1. Make a small change to `app.py` (like adding a comment)
2. Commit and push to GitHub
3. Streamlit Cloud will automatically redeploy
4. The cache will be cleared

### Method 4: Manual Cache Clear (For Local Testing)

If testing locally:

1. Click **"☰" (hamburger menu)** in Streamlit app
2. Click **"Clear cache"**
3. Click **"Rerun"**

---

## Verify Weights Are Correct

After clearing cache, verify:

1. The sidebar should show: **"XGBoost (50% weight), CatBoost (50% weight)"**
2. The "Model Details & Breakdown" section should show: **"50% XGBoost + 50% CatBoost"**
3. Check `models/ensemble_weights.json` file on GitHub - it should show `{"w_xgb": 0.5, "w_cat": 0.5}`

---

## Current Status

- ✅ **GitHub Repository**: Weights file is 50/50
- ✅ **Local File**: Weights file is 50/50
- ⚠️ **Streamlit Cloud**: May be using cached 20/80 (needs cache clear)

---

## Quick Fix

**On Streamlit Cloud:**
1. Go to app dashboard
2. Click "Clear cache" or "Reboot app"
3. Wait for restart
4. Check weights are now 50/50

---

**The weights file is correct (50/50) - it's just a cache issue!**

