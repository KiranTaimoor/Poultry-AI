"""
Poultry AI Agent API v3 — With JAPFA Standards
Deploy to Railway/Render. Include poultry_model_v3.pkl in same folder.
"""
from fastapi import UploadFile, File
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np, pandas as pd, joblib, os
from datetime import datetime

app = FastAPI(title="Poultry AI v3", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
API_KEY = os.environ.get("POULTRY_API_KEY", "poultry-ai-2025")
artifact = None
latest_predictions = {"predictions": [], "summary": {}, "actions": []}

class Record(BaseModel):
    Farm: str = ""; Shed: str = ""; Date: str = ""; Hour: int = 0
    Temp: float = 0; Humidity: float = 0; nh3: float = 0
    ph: Optional[float] = None; co2: Optional[float] = None; tds: Optional[float] = None
    DOC: float = 0; Weight: float = 0
    Water_Consumption: float = 0; Mortality: float = 0
    flock_id: str = ""; flock_age_days: int = 1

class PredictRequest(BaseModel):
    records: List[Record]

def verify(x_api_key: str = Header(None)):
    if x_api_key != API_KEY: raise HTTPException(401, "Invalid API key")

def engineer_features(df, feature_cols):
    """Same feature engineering as training."""
    for col in ['ph','co2','tds','Weight','Water Consumption','Mortality']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
    df['Location'] = df['Farm'].astype(str) + '|' + df['Shed'].astype(str)
    if 'flock_age_days' not in df.columns or df['flock_age_days'].max() == 0:
        fstart = df.groupby('flock_id')['Date'].transform('min')
        df['flock_age_days'] = (df['Date'] - fstart).dt.days + 1

    daily = df.groupby(['Farm','Shed','Location','Date']).agg(
        avg_temp=('Temp','mean'),max_temp=('Temp','max'),min_temp=('Temp','min'),std_temp=('Temp','std'),
        avg_humidity=('Humidity','mean'),max_humidity=('Humidity','max'),min_humidity=('Humidity','min'),std_humidity=('Humidity','std'),
        avg_nh3=('nh3','mean'),max_nh3=('nh3','max'),std_nh3=('nh3','std'),min_nh3=('nh3','min'),
        avg_ph=('ph','mean'),avg_co2=('co2','mean'),avg_tds=('tds','mean'),
        doc_weight=('DOC','first'),avg_weight=('Weight','mean'),max_weight=('Weight','max'),
        water_consumption=('Water Consumption','mean'),daily_mortality=('Mortality','sum'),
        flock_age=('flock_age_days','first'),readings=('Temp','count'),
    ).reset_index()
    for c in ['avg_ph','avg_co2','avg_tds','std_temp','std_humidity','std_nh3','min_nh3']:
        daily[c] = daily[c].fillna(0)
    daily['daily_mortality'] = daily['daily_mortality'].fillna(0)
    daily['water_consumption'] = daily['water_consumption'].fillna(0)
    daily['avg_weight'] = daily['avg_weight'].fillna(0)
    daily['doc_weight'] = daily['doc_weight'].fillna(42)
    daily = daily.sort_values(['Location','Date']).reset_index(drop=True)

    # All engineered features
    daily['temp_range'] = daily['max_temp'] - daily['min_temp']
    daily['humidity_range'] = daily['max_humidity'] - daily['min_humidity']
    daily['nh3_range'] = daily.get('max_nh3',0) - daily.get('min_nh3',0) if 'min_nh3' in daily.columns else 0
    daily['heat_stress'] = (daily['avg_temp']-25).clip(lower=0)*daily['avg_humidity']/100
    daily['cold_stress'] = (18-daily['avg_temp']).clip(lower=0)
    daily['ventilation_stress'] = daily['avg_nh3']*daily['avg_humidity']/100
    daily['temp_instability'] = daily['std_temp']/daily['avg_temp'].clip(lower=1)
    daily['humidity_instability'] = daily['std_humidity']/daily['avg_humidity'].clip(lower=1)
    dm = daily['doc_weight'].median()
    daily['doc_deviation'] = abs(daily['doc_weight']-dm)
    daily['doc_below_avg'] = (daily['doc_weight']<dm).astype(int)
    daily['weight_gain_vs_doc'] = daily['avg_weight']-daily['doc_weight']
    daily['weight_to_age_ratio'] = daily['avg_weight']/daily['flock_age'].clip(lower=1)
    daily['water_per_weight'] = daily['water_consumption']/daily['avg_weight'].clip(lower=1)
    daily['water_low'] = (daily['water_consumption']<daily['water_consumption'].quantile(0.1)).astype(int) if len(daily)>10 else 0
    daily['air_quality'] = daily['avg_nh3']*0.5+daily['avg_co2']*0.3+daily['avg_tds']*0.2
    daily['nh3_danger'] = (daily['avg_nh3']>20).astype(int)
    daily['temp_x_humidity'] = daily['avg_temp']*daily['avg_humidity']
    daily['temp_x_nh3'] = daily['avg_temp']*daily['avg_nh3']
    daily['doc_x_temp'] = daily['doc_weight']*daily['avg_temp']
    daily['temp_breach'] = 0; daily['humidity_breach'] = 0; daily['nh3_breach'] = 0
    daily['temp_vs_standard'] = 0; daily['weight_vs_standard'] = 0
    daily['total_breach_score'] = 0; daily['breach_x_heat'] = 0
    daily['weight_x_nh3'] = daily['avg_weight']*daily['avg_nh3']
    daily['doc_category'] = 2
    daily['nh3_warning'] = (daily['avg_nh3']>10).astype(int)
    daily['humidity_x_nh3'] = daily['avg_humidity']*daily['avg_nh3']
    daily['doc_x_nh3'] = daily['doc_weight']*daily['avg_nh3']
    daily['heat_x_nh3'] = daily['heat_stress']*daily['avg_nh3']
    daily['day_of_week'] = daily['Date'].dt.dayofweek
    daily['month'] = daily['Date'].dt.month
    daily['day_of_year'] = daily['Date'].dt.dayofyear
    daily['is_summer'] = daily['month'].isin([5,6,7,8,9]).astype(int)
    daily['is_winter'] = daily['month'].isin([11,12,1,2]).astype(int)
    daily['is_weekend'] = (daily['day_of_week']>=5).astype(int)
    daily['quarter'] = daily['Date'].dt.quarter
    daily['flock_age_weeks'] = daily['flock_age']//7
    daily['is_first_week'] = (daily['flock_age']<=7).astype(int)
    daily['is_first_2weeks'] = (daily['flock_age']<=14).astype(int)
    daily['is_finisher'] = (daily['flock_age']>28).astype(int)
    for w in [3,7,14]:
        for c in ['avg_temp','avg_humidity','avg_nh3','daily_mortality','avg_weight']:
            daily[f'{c}_roll{w}'] = daily.groupby('Location')[c].transform(lambda x: x.rolling(w,min_periods=1).mean())
    for c in ['avg_temp','avg_humidity','avg_nh3']:
        daily[f'{c}_ema7'] = daily.groupby('Location')[c].transform(lambda x: x.ewm(span=7,min_periods=1).mean())
    for lag in [1,2,3]:
        for c in ['avg_temp','avg_humidity','avg_nh3','daily_mortality','total_breach_score']:
            daily[f'{c}_lag{lag}'] = daily.groupby('Location')[c].shift(lag).fillna(0)
    daily['temp_change'] = daily['avg_temp']-daily['avg_temp_lag1']
    daily['humidity_change'] = daily['avg_humidity']-daily['avg_humidity_lag1']
    daily['nh3_change'] = daily['avg_nh3']-daily['avg_nh3_lag1']
    daily['temp_shock'] = abs(daily['temp_change'])
    daily['humidity_shock'] = abs(daily['humidity_change'])
    daily['consecutive_breach'] = 0
    daily['cumulative_mort_7d'] = daily.groupby('Location')['daily_mortality'].transform(
        lambda x: x.rolling(7,min_periods=1).sum())-daily['daily_mortality']

    # Encode
    enc = artifact
    daily['farm_encoded'] = daily['Farm'].apply(lambda x: enc['farm_encoder'].transform([x])[0] if x in enc['farm_encoder'].classes_ else -1)
    daily['shed_encoded'] = daily['Shed'].apply(lambda x: enc['shed_encoder'].transform([x])[0] if x in enc['shed_encoder'].classes_ else -1)
    daily['location_encoded'] = daily['Location'].apply(lambda x: enc['location_encoder'].transform([x])[0] if x in enc['location_encoder'].classes_ else -1)

    # Ensure all features exist
    for f in feature_cols:
        if f not in daily.columns: daily[f] = 0
    return daily

def generate_actions(row):
    """Generate actionable recommendations based on predictions and breaches."""
    actions = []
    if row.get('avg_temp',0) > 30:
        actions.append("🌡️ URGENT: Activate pad cooling and increase fan speed")
    if row.get('avg_temp',0) < 18:
        actions.append("❄️ URGENT: Activate heating system")
    if row.get('avg_nh3',0) > 15:
        actions.append("🧪 Increase ventilation — NH3 above safe levels")
    if row.get('avg_nh3',0) > 20:
        actions.append("🚨 EMERGENCY: NH3 critical — maximum ventilation + check litter")
    if row.get('avg_humidity',0) > 75:
        actions.append("💧 Reduce humidity — check foggers and ventilation rate")
    if row.get('avg_humidity',0) < 40:
        actions.append("💧 Increase humidity — activate foggers")
    if row.get('water_consumption',0) > 0 and row.get('water_low',0) == 1:
        actions.append("🚰 Water intake low — check water lines, pressure, nipples")
    if row.get('heat_stress',0) > 5:
        actions.append("🔥 Heat stress detected — implement ventilation management")
    if row.get('weight_vs_standard',0) < -50:
        actions.append("⚖️ Weight below standard — review feed quality and intake")
    if not actions:
        actions.append("✅ All parameters within acceptable range")
    return actions

@app.on_event("startup")
def load():
    global artifact
    if os.path.exists('poultry_model_v3.pkl'):
        artifact = joblib.load('poultry_model_v3.pkl')
        print(f"✅ Model v3 loaded ({artifact['n_samples']} samples)")

@app.get("/health")
def health():
    return {"status":"healthy" if artifact else "no_model","model_loaded":artifact is not None}
    
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...), x_api_key: str = Header(None)):
    verify(x_api_key)
    if not artifact:
        raise HTTPException(503, "No model")
    
    import io
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    df.columns = df.columns.str.strip()
    
    # Limit to 5000 rows to prevent timeout
    if len(df) > 5000:
        df = df.tail(5000)
    
    # Convert to records format
    records = []
    for _, r in df.iterrows():
        records.append(Record(
            Farm=str(r.get('Farm', '')),
            Shed=str(r.get('Shed', '')),
            Date=str(r.get('Date', '')),
            Hour=int(r.get('Hour', 0)),
            Temp=float(r.get('Temp', 0)),
            Humidity=float(r.get('Humidity', 0)),
            nh3=float(r.get('nh3', 0)) if r.get('nh3') != '-' else 0,
            ph=float(r.get('ph', 0)) if r.get('ph') not in ['-', None, ''] else None,
            co2=float(r.get('co2', 0)) if r.get('co2') not in ['-', None, ''] else None,
            tds=float(r.get('tds', 0)) if r.get('tds') not in ['-', None, ''] else None,
            DOC=float(r.get('DOC', 0)),
            Weight=float(r.get('Weight', 0)),
            Water_Consumption=float(r.get('Water Consumption', 0)),
            Mortality=float(r.get('Mortality', 0)),
            flock_id=str(r.get('flock.id', '')),
            flock_age_days=1,
        ))
    
    # Use existing predict logic
    data = PredictRequest(records=records)
    return predict(data, x_api_key)
@app.post("/predict")
def predict(data: PredictRequest, x_api_key: str = Header(None)):
    verify(x_api_key)
    if not artifact: raise HTTPException(503,"No model")
    global latest_predictions
    records = []
    for r in data.records:
        d = r.model_dump()
        d['Water Consumption'] = d.pop('Water_Consumption', 0)
        d['flock.id'] = d.pop('flock_id', '')
        records.append(d)
    df = pd.DataFrame(records)
    fc = artifact['feature_names']
    daily = engineer_features(df, fc)
    X = daily[fc].fillna(0).values
    Xs = artifact['scaler'].transform(X)

    if artifact['best_method'] == 'two_stage':
        prob = artifact['classifier'].predict_proba(Xs)[:,1]
        reg_p = np.expm1(artifact['regressor'].predict(Xs)).clip(min=0)
        preds = np.where(prob>0.3, reg_p*prob, 0)
    else:
        preds = artifact['single_model'].predict(Xs).clip(min=0)

    results = []
    for i,(_, row) in enumerate(daily.iterrows()):
        p = round(float(max(0, preds[i])), 2)
        risk = "CRITICAL" if p>10 else "HIGH" if p>5 else "MEDIUM" if p>2 else "LOW"
        acts = generate_actions(row.to_dict())
        results.append({
            "Farm":row["Farm"],"Shed":row.get("Shed",""),"Date":str(row["Date"].date()),
            "predicted_mortality":p,"actual_mortality":round(float(row["daily_mortality"]),2),
            "risk_level":risk,"avg_temp":round(float(row["avg_temp"]),1),
            "avg_humidity":round(float(row["avg_humidity"]),1),"avg_nh3":round(float(row["avg_nh3"]),2),
            "heat_stress":round(float(row.get("heat_stress",0)),2),
            "weight":round(float(row.get("avg_weight",0)),0),
            "water":round(float(row.get("water_consumption",0)),0),
            "actions":acts,
        })
    rdf = pd.DataFrame(results)
    summary = {
        "records":len(results),"avg_mortality":round(rdf["predicted_mortality"].mean(),2),
        "critical":int((rdf["risk_level"]=="CRITICAL").sum()),
        "high":int((rdf["risk_level"]=="HIGH").sum()),
        "farms":rdf["Farm"].unique().tolist(),
    }
    all_actions = list(set(a for r in results for a in r["actions"] if not a.startswith("✅")))
    latest_predictions = {"predictions":results,"summary":summary,"actions":all_actions}
    return latest_predictions

@app.get("/results")
def results(x_api_key:str=Header(None)):
    verify(x_api_key); return latest_predictions

@app.get("/results/csv")
def csv(x_api_key:str=Header(None)):
    verify(x_api_key)
    if not latest_predictions["predictions"]:
        return PlainTextResponse("Farm,Shed,Date,predicted_mortality,risk_level\n")
    lines=["Farm,Shed,Date,predicted_mortality,risk_level,avg_temp,avg_humidity,avg_nh3,actions"]
    for p in latest_predictions["predictions"]:
        acts = "; ".join(p["actions"])
        lines.append(f"{p['Farm']},{p['Shed']},{p['Date']},{p['predicted_mortality']},{p['risk_level']},{p['avg_temp']},{p['avg_humidity']},{p['avg_nh3']},\"{acts}\"")
    return PlainTextResponse("\n".join(lines),media_type="text/csv")

@app.get("/metrics")
def metrics(x_api_key:str=Header(None)):
    verify(x_api_key); return artifact["metrics"] if artifact else {}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("PORT",8000)))
