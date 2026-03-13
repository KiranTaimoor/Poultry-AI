# Poultry AI Agent v3 — Step by Step Guide

## Overview

```
INPUT:                        AUTO-SYNC:                    OUTPUT:
Temperature                   Baseline mortality/day        Daily mortality forecast
Ammonia (NH3)                 JAPFA standard breaches       Cumulative mortality by final day
CO2                           Weight vs standard            Risk level (LOW/MED/HIGH/CRITICAL)
Humidity                      Water vs standard             Actions to take
Water intake                                                (ventilation, water quality, etc.)
Weight
DOC
```

## Step 1: Upload Files to Google Drive

Put these files in your Google Drive (root or any folder):
- `2023.xlsx`
- `2024.xlsx`
- `2025.xlsx`
- `2026.xlsx`
- `JAPFA_Standards_-_All_Parameters_Ranges__1_.xlsx`

## Step 2: Train Model in Google Colab

1. Open https://colab.research.google.com → New Notebook
2. Copy each CELL from `poultry_v3_complete.py` into separate cells
3. Run cells 1-11 with Shift+Enter
4. Cell 11 downloads `poultry_model_v3.pkl` to your computer
5. Model is also saved in Google Drive as backup

## Step 3: Deploy FastAPI

### Option A: Railway (easiest)
```bash
# Put these files in a GitHub repo:
#   main.py, requirements.txt, Dockerfile, poultry_model_v3.pkl
git init && git add . && git commit -m "poultry ai v3"
git remote add origin https://github.com/YOU/poultry-ai.git
git push -u origin main
# Go to railway.app → New → Deploy from GitHub
# Set env: POULTRY_API_KEY=poultry-ai-2025
```

### Option B: Render
1. Push same files to GitHub
2. render.com → New Web Service → Connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option C: Docker (your server)
```bash
docker build -t poultry-ai .
docker run -d -p 8000:8000 -e POULTRY_API_KEY=poultry-ai-2025 poultry-ai
```

## Step 4: Test the API

```bash
curl https://YOUR-API-URL/health
# Should return: {"status":"healthy","model_loaded":true}
```

## Step 5: Connect n8n

Use the working Qlik export pattern we proved:
1. Export last 7 days from Qlik using Reports API
2. POST data to your ML API /predict
3. Build HTML report from response
4. Send via Gmail
5. Store results at /results/csv for Qlik to read back

## API Endpoints

| Endpoint | Method | What |
|----------|--------|------|
| /health | GET | Is API running? |
| /predict | POST | Send data, get predictions + actions |
| /results | GET | Latest predictions (JSON) |
| /results/csv | GET | Latest predictions (CSV for Qlik) |
| /metrics | GET | Model performance stats |
