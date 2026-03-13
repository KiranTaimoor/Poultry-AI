"""
═══════════════════════════════════════════════════════════════
POULTRY AI v3 — COMPLETE TRAINING WITH JAPFA STANDARDS
═══════════════════════════════════════════════════════════════

New in v3:
  - JAPFA Standards integration (Good/Poor/Bad ranges for each day)
  - Weight impact analysis
  - Water Consumption analysis
  - Automatic breach detection (temp, humidity, NH3, CO2 vs standards)
  - Flock age tracking (days since placement)
  - Two-stage LightGBM model
  - Actionable recommendations

Data Fields:
  Farm, Shed, flock.id, Date, Hour
  Temp, Humidity, nh3, ph, co2, tds
  Weight, Water Consumption, DOC, Mortality

Upload JAPFA_Standards_-_All_Parameters_Ranges__1_.xlsx to Drive too!

Copy each CELL between ═══ lines into separate Colab cells.
═══════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Setup + Mount Drive
# ═══════════════════════════════════════════════════════════════

!pip install openpyxl lightgbm shap -q

from google.colab import drive
drive.mount('/content/drive')

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Find data folder
data_folder = None
for root, dirs, flist in os.walk('/content/drive/MyDrive'):
    if '2023.xlsx' in [f.lower() for f in flist]:
        data_folder = root
        break
print(f"📁 Data folder: {data_folder}" if data_folder else "⚠️ Set FOLDER_PATH in Cell 2")


# ═══════════════════════════════════════════════════════════════
# CELL 2: Load All Data + JAPFA Standards
# ═══════════════════════════════════════════════════════════════

FOLDER_PATH = data_folder or '/content/drive/MyDrive/'

# Load yearly files
dfs = []
for yr in ['2023','2024','2025','2026']:
    path = os.path.join(FOLDER_PATH, f'{yr}.xlsx')
    if os.path.exists(path):
        print(f"Loading {yr}.xlsx...", end=" ")
        t = pd.read_excel(path)
        print(f"✅ {len(t):,} rows")
        dfs.append(t)
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()
print(f"\n📊 Total: {len(df):,} rows | Columns: {list(df.columns)}")

# Load JAPFA Standards
std_path = None
for root, dirs, flist in os.walk('/content/drive/MyDrive'):
    for f in flist:
        if 'japfa' in f.lower() and f.endswith('.xlsx'):
            std_path = os.path.join(root, f)
            break

if std_path:
    print(f"\n📋 Loading JAPFA Standards from: {std_path}")
    std_xls = pd.ExcelFile(std_path)

    # Parse Temp standards
    temp_std = pd.read_excel(std_xls, 'Temp Standards', skiprows=2)
    temp_std.columns = ['Day','SetPoint','GoodMin','GoodMax','PoorMin','PoorMax','BadMin','BadMax']
    temp_std = temp_std.dropna(subset=['Day']).reset_index(drop=True)
    temp_std['Day'] = temp_std['Day'].astype(int)

    # Parse Humidity standards
    hum_std = pd.read_excel(std_xls, 'Humidity Standards', skiprows=2)
    hum_std.columns = ['Day','SetPoint','GoodMin','GoodMax','PoorMin','PoorMax','BadMin','BadMax']
    hum_std = hum_std.dropna(subset=['Day']).reset_index(drop=True)
    hum_std['Day'] = hum_std['Day'].astype(int)

    # Parse NH3 standards
    nh3_std = pd.read_excel(std_xls, 'Ammonia Standards', skiprows=2)
    nh3_std.columns = ['Day','SetPoint','GoodMin','GoodMax','PoorMin','PoorMax','BadMin','BadMax']
    nh3_std = nh3_std.dropna(subset=['Day']).reset_index(drop=True)
    nh3_std['Day'] = nh3_std['Day'].astype(int)

    # Parse Daily Weight standards
    wt_std = pd.read_excel(std_xls, 'Daily Weight Standards', skiprows=3)
    wt_std.columns = ['Day','Standard','From','To']
    wt_std = wt_std.dropna(subset=['Day']).reset_index(drop=True)
    wt_std['Day'] = wt_std['Day'].astype(int)
    wt_std['Standard'] = pd.to_numeric(wt_std['Standard'], errors='coerce')

    # Parse Daily Mortality standards
    mort_std = pd.read_excel(std_xls, 'Daily Mortality Standards', skiprows=2)
    mort_std.columns = ['Day','Standard','GoodMin','GoodMax','PoorMin','PoorMax','BadMin','BadMax']
    mort_std = mort_std.dropna(subset=['Day']).reset_index(drop=True)
    mort_std['Day'] = mort_std['Day'].astype(int)
    mort_std['Standard'] = pd.to_numeric(mort_std['Standard'], errors='coerce')

    # Parse Water standards
    water_std = pd.read_excel(std_xls, 'Daily Water mlday standards', skiprows=2)
    water_std.columns = ['Day','Standard','GoodMin','GoodMax','PoorMin','PoorMax','BadMin','BadMax']
    water_std = water_std.dropna(subset=['Day']).reset_index(drop=True)
    water_std['Day'] = water_std['Day'].astype(int)
    water_std['Standard'] = pd.to_numeric(water_std['Standard'], errors='coerce')

    print(f"  ✅ Loaded standards for: Temp, Humidity, NH3, Weight, Mortality, Water")
    print(f"  Day range: 1 to {temp_std['Day'].max()}")
else:
    print("⚠️ JAPFA Standards file not found. Upload it to Google Drive.")
    temp_std = hum_std = nh3_std = wt_std = mort_std = water_std = None


# ═══════════════════════════════════════════════════════════════
# CELL 3: Data Cleaning
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("🧹 DATA CLEANING")
print("=" * 60)

# Convert numeric
for col in ['Temp','Humidity','nh3','ph','co2','tds','DOC','Mortality','Weight','Water Consumption']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        valid = df[col].notna().sum()
        pct = df[col].isna().sum()/len(df)*100
        print(f"  {col:25s}: {valid:>10,} valid ({pct:5.1f}% missing)")

df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
df['Location'] = df['Farm'].astype(str) + '|' + df['Shed'].astype(str)

# Compute flock age: days since first record for each flock
flock_start = df.groupby('flock.id')['Date'].transform('min')
df['flock_age_days'] = (df['Date'] - flock_start).dt.days + 1  # day 1 = placement day

print(f"\n📅 Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"🏠 Farms: {sorted(df['Farm'].unique().tolist())}")
print(f"🐔 Flocks: {df['flock.id'].nunique()}")
print(f"🐣 Flock age range: day {df['flock_age_days'].min()} to day {df['flock_age_days'].max()}")


# ═══════════════════════════════════════════════════════════════
# CELL 4: Daily Aggregation + JAPFA Breach Detection
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("📦 DAILY AGGREGATION + BREACH DETECTION")
print("=" * 60)

daily = df.groupby(['Farm','Shed','Location','Date','flock.id']).agg(
    avg_temp=('Temp','mean'), max_temp=('Temp','max'), min_temp=('Temp','min'), std_temp=('Temp','std'),
    avg_humidity=('Humidity','mean'), max_humidity=('Humidity','max'), min_humidity=('Humidity','min'), std_humidity=('Humidity','std'),
    avg_nh3=('nh3','mean'), max_nh3=('nh3','max'), std_nh3=('nh3','std'),
    avg_ph=('ph','mean'), avg_co2=('co2','mean'), avg_tds=('tds','mean'),
    doc_weight=('DOC','first'),
    avg_weight=('Weight','mean'), max_weight=('Weight','max'),
    water_consumption=('Water Consumption','mean'),
    daily_mortality=('Mortality','sum'),
    flock_age=('flock_age_days','first'),
    readings=('Temp','count'),
).reset_index()

# Fill NaN
for col in ['avg_ph','avg_co2','avg_tds','std_temp','std_humidity','std_nh3']:
    daily[col] = daily[col].fillna(0)
daily['doc_weight'] = daily['doc_weight'].fillna(daily['doc_weight'].median())
daily['daily_mortality'] = daily['daily_mortality'].fillna(0)
daily['water_consumption'] = daily['water_consumption'].fillna(0)
daily['avg_weight'] = daily['avg_weight'].fillna(0)

# ── JAPFA Standards Breach Detection ──────────────────────────
if temp_std is not None:
    # Map standards by flock age (day)
    std_map_temp = temp_std.set_index('Day')
    std_map_hum = hum_std.set_index('Day')
    std_map_nh3 = nh3_std.set_index('Day')
    std_map_wt = wt_std.set_index('Day')
    std_map_mort = mort_std.set_index('Day')

    def get_breach_level(value, day, std_df, col_prefix=''):
        """Returns: 0=Good, 1=Poor, 2=Bad, 3=Critical"""
        if day not in std_df.index or pd.isna(value):
            return 0
        row = std_df.loc[day]
        good_min = pd.to_numeric(row.get('GoodMin', 0), errors='coerce') or 0
        good_max = pd.to_numeric(row.get('GoodMax', 999), errors='coerce') or 999
        poor_min = pd.to_numeric(row.get('PoorMin', 0), errors='coerce') or 0
        poor_max = pd.to_numeric(row.get('PoorMax', 999), errors='coerce') or 999
        bad_min = pd.to_numeric(row.get('BadMin', 0), errors='coerce') or 0
        bad_max = pd.to_numeric(row.get('BadMax', 999), errors='coerce') or 999

        if good_min <= value <= good_max:
            return 0  # Good
        elif poor_min <= value <= poor_max:
            return 1  # Poor
        elif bad_min <= value <= bad_max:
            return 2  # Bad
        else:
            return 3  # Critical

    # Clamp flock age to available standard days (1-35 or 1-38)
    max_std_day = int(temp_std['Day'].max())
    daily['std_day'] = daily['flock_age'].clip(1, max_std_day)

    # Calculate breach levels
    daily['temp_breach'] = daily.apply(lambda r: get_breach_level(r['avg_temp'], r['std_day'], std_map_temp), axis=1)
    daily['humidity_breach'] = daily.apply(lambda r: get_breach_level(r['avg_humidity'], r['std_day'], std_map_hum), axis=1)
    daily['nh3_breach'] = daily.apply(lambda r: get_breach_level(r['avg_nh3'], r['std_day'], std_map_nh3), axis=1)

    # Temperature deviation from JAPFA standard setpoint
    daily['temp_vs_standard'] = daily.apply(
        lambda r: r['avg_temp'] - pd.to_numeric(std_map_temp.loc[r['std_day'],'GoodMin'], errors='coerce')
        if r['std_day'] in std_map_temp.index else 0, axis=1
    )

    # Weight deviation from standard
    daily['weight_vs_standard'] = daily.apply(
        lambda r: r['avg_weight'] - pd.to_numeric(std_map_wt.loc[r['std_day'],'Standard'], errors='coerce')
        if r['std_day'] in std_map_wt.index and r['avg_weight'] > 0 else 0, axis=1
    )

    # Total breaches score
    daily['total_breach_score'] = daily['temp_breach'] + daily['humidity_breach'] + daily['nh3_breach']

    print(f"\n🚨 BREACH SUMMARY:")
    print(f"   Temp breaches:     {(daily['temp_breach'] > 0).sum():,} days ({(daily['temp_breach']>0).sum()/len(daily)*100:.1f}%)")
    print(f"   Humidity breaches: {(daily['humidity_breach'] > 0).sum():,} days ({(daily['humidity_breach']>0).sum()/len(daily)*100:.1f}%)")
    print(f"   NH3 breaches:      {(daily['nh3_breach'] > 0).sum():,} days ({(daily['nh3_breach']>0).sum()/len(daily)*100:.1f}%)")
    print(f"   Any breach:        {(daily['total_breach_score'] > 0).sum():,} days ({(daily['total_breach_score']>0).sum()/len(daily)*100:.1f}%)")
else:
    daily['temp_breach'] = 0
    daily['humidity_breach'] = 0
    daily['nh3_breach'] = 0
    daily['temp_vs_standard'] = 0
    daily['weight_vs_standard'] = 0
    daily['total_breach_score'] = 0
    daily['std_day'] = daily['flock_age'].clip(1, 35)

print(f"\n✅ {len(df):,} hourly → {len(daily):,} daily records")


# ═══════════════════════════════════════════════════════════════
# CELL 5: Feature Engineering (70+ features)
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("⚙️ FEATURE ENGINEERING — 70+ Features")
print("=" * 60)

daily = daily.sort_values(['Location','Date']).reset_index(drop=True)

# ── Stress ────────────────────────────────────────────────────
daily['temp_range'] = daily['max_temp'] - daily['min_temp']
daily['humidity_range'] = daily['max_humidity'] - daily['min_humidity']
daily['heat_stress'] = (daily['avg_temp'] - 25).clip(lower=0) * daily['avg_humidity'] / 100
daily['cold_stress'] = (18 - daily['avg_temp']).clip(lower=0)
daily['ventilation_stress'] = daily['avg_nh3'] * daily['avg_humidity'] / 100
daily['temp_instability'] = daily['std_temp'] / daily['avg_temp'].clip(lower=1)
daily['humidity_instability'] = daily['std_humidity'] / daily['avg_humidity'].clip(lower=1)

# ── DOC + Weight ──────────────────────────────────────────────
doc_median = daily['doc_weight'].median()
daily['doc_deviation'] = abs(daily['doc_weight'] - doc_median)
daily['doc_below_avg'] = (daily['doc_weight'] < doc_median).astype(int)
daily['weight_gain_vs_doc'] = daily['avg_weight'] - daily['doc_weight']
daily['weight_to_age_ratio'] = daily['avg_weight'] / daily['flock_age'].clip(lower=1)

# ── Water ─────────────────────────────────────────────────────
daily['water_per_weight'] = daily['water_consumption'] / daily['avg_weight'].clip(lower=1)
daily['water_low'] = (daily['water_consumption'] < daily['water_consumption'].quantile(0.1)).astype(int)

# ── Air quality ───────────────────────────────────────────────
daily['air_quality'] = daily['avg_nh3']*0.5 + daily['avg_co2']*0.3 + daily['avg_tds']*0.2
daily['nh3_danger'] = (daily['avg_nh3'] > 20).astype(int)

# ── Interactions ──────────────────────────────────────────────
daily['temp_x_humidity'] = daily['avg_temp'] * daily['avg_humidity']
daily['temp_x_nh3'] = daily['avg_temp'] * daily['avg_nh3']
daily['doc_x_temp'] = daily['doc_weight'] * daily['avg_temp']
daily['breach_x_heat'] = daily['total_breach_score'] * daily['heat_stress']
daily['weight_x_nh3'] = daily['avg_weight'] * daily['avg_nh3']

# ── Time ──────────────────────────────────────────────────────
daily['day_of_week'] = daily['Date'].dt.dayofweek
daily['month'] = daily['Date'].dt.month
daily['day_of_year'] = daily['Date'].dt.dayofyear
daily['is_summer'] = daily['month'].isin([5,6,7,8,9]).astype(int)
daily['is_winter'] = daily['month'].isin([11,12,1,2]).astype(int)

# ── Flock age features ────────────────────────────────────────
daily['flock_age_weeks'] = daily['flock_age'] // 7
daily['is_first_week'] = (daily['flock_age'] <= 7).astype(int)
daily['is_first_2weeks'] = (daily['flock_age'] <= 14).astype(int)
daily['is_finisher'] = (daily['flock_age'] > 28).astype(int)

# ── Rolling (3, 7, 14 day) ───────────────────────────────────
for w in [3, 7, 14]:
    for col in ['avg_temp','avg_humidity','avg_nh3','daily_mortality','avg_weight']:
        daily[f'{col}_roll{w}'] = daily.groupby('Location')[col].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )

# EMA
for col in ['avg_temp','avg_humidity','avg_nh3']:
    daily[f'{col}_ema7'] = daily.groupby('Location')[col].transform(
        lambda x: x.ewm(span=7, min_periods=1).mean()
    )

# ── Lags (1, 2, 3 day) ───────────────────────────────────────
for lag in [1, 2, 3]:
    for col in ['avg_temp','avg_humidity','avg_nh3','daily_mortality','total_breach_score']:
        daily[f'{col}_lag{lag}'] = daily.groupby('Location')[col].shift(lag).fillna(0)

# ── Changes / Shocks ─────────────────────────────────────────
daily['temp_change'] = daily['avg_temp'] - daily['avg_temp_lag1']
daily['humidity_change'] = daily['avg_humidity'] - daily['avg_humidity_lag1']
daily['nh3_change'] = daily['avg_nh3'] - daily['avg_nh3_lag1']
daily['temp_shock'] = abs(daily['temp_change'])
daily['humidity_shock'] = abs(daily['humidity_change'])

# ── Cumulative ────────────────────────────────────────────────
daily['consecutive_breach'] = daily.groupby('Location')['total_breach_score'].transform(
    lambda x: x.gt(0).groupby((~x.gt(0)).cumsum()).cumsum()
)
daily['cumulative_mort_7d'] = daily.groupby('Location')['daily_mortality'].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
) - daily['daily_mortality']

# ── Encode ────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
farm_enc = LabelEncoder(); daily['farm_encoded'] = farm_enc.fit_transform(daily['Farm'])
shed_enc = LabelEncoder(); daily['shed_encoded'] = shed_enc.fit_transform(daily['Shed'])
loc_enc = LabelEncoder(); daily['location_encoded'] = loc_enc.fit_transform(daily['Location'])

# ── Feature list ──────────────────────────────────────────────
feature_cols = [c for c in daily.columns if c not in [
    'Farm','Shed','Location','Date','flock.id','daily_mortality','std_day'
] and daily[c].dtype in ['float64','int64','int32','float32']]

print(f"\n✅ Total features: {len(feature_cols)}")
print(f"   Categories: Environment, Stress, DOC/Weight, Water, JAPFA Breaches,")
print(f"   Interactions, Time, Flock Age, Rolling, Lags, Changes, Cumulative")


# ═══════════════════════════════════════════════════════════════
# CELL 6: Two-Stage LightGBM Training
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("🧠 TWO-STAGE LightGBM TRAINING")
print("=" * 60)

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, f1_score,
                              classification_report, precision_score, recall_score)
import time

X = daily[feature_cols].fillna(0).values
y = daily['daily_mortality'].values
y_bin = (y > 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-based split
split = int(len(X) * 0.8)
Xtr, Xte = X_scaled[:split], X_scaled[split:]
ytr, yte = y[:split], y[split:]
ytr_bin, yte_bin = y_bin[:split], y_bin[split:]

print(f"Train: {len(Xtr):,} | Test: {len(Xte):,}")
print(f"Mortality rate (train): {ytr_bin.mean()*100:.1f}% | (test): {yte_bin.mean()*100:.1f}%")

# ── STAGE 1: Classifier ──────────────────────────────────────
print(f"\n{'─'*60}")
print("STAGE 1: Will mortality occur today?")

start = time.time()
clf = lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.02, max_depth=8, num_leaves=127,
    min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.2, class_weight='balanced',
    random_state=42, verbose=-1,
)
clf.fit(Xtr, ytr_bin)
clf_pred = clf.predict(Xte)
clf_prob = clf.predict_proba(Xte)[:, 1]

print(f"\n  Accuracy:  {accuracy_score(yte_bin, clf_pred):.4f}")
print(f"  Precision: {precision_score(yte_bin, clf_pred):.4f}")
print(f"  Recall:    {recall_score(yte_bin, clf_pred):.4f}")
print(f"  F1:        {f1_score(yte_bin, clf_pred):.4f}")

# ── STAGE 2: Regressor (on positive only) ─────────────────────
print(f"\n{'─'*60}")
print("STAGE 2: How many deaths? (positive days only)")

pos_mask = ytr > 0
Xtr_pos = Xtr[pos_mask]
ytr_pos_log = np.log1p(ytr[pos_mask])

reg = lgb.LGBMRegressor(
    n_estimators=800, learning_rate=0.02, max_depth=8, num_leaves=127,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.2, random_state=42, verbose=-1,
)
reg.fit(Xtr_pos, ytr_pos_log)

# ── Combined prediction ───────────────────────────────────────
reg_pred_log = reg.predict(Xte)
reg_pred = np.expm1(reg_pred_log).clip(min=0)
combined = np.where(clf_prob > 0.3, reg_pred * clf_prob, 0)

# ── Also train single model for comparison ────────────────────
single = lgb.LGBMRegressor(
    n_estimators=800, learning_rate=0.02, max_depth=8, num_leaves=127,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.7,
    random_state=42, verbose=-1,
)
single.fit(Xtr, ytr)
single_pred = single.predict(Xte).clip(min=0)

duration = time.time() - start

# ── Results ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"📊 RESULTS COMPARISON")
print(f"{'='*60}")
print(f"\n  {'Method':<30s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
print(f"  {'─'*55}")

for name, pred in [('Single LightGBM', single_pred), ('Two-Stage Combined', combined)]:
    mae = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2 = r2_score(yte, pred)
    print(f"  {name:<30s} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

# Pick best
best_pred = combined if r2_score(yte, combined) >= r2_score(yte, single_pred) else single_pred
best_name = 'two_stage' if r2_score(yte, combined) >= r2_score(yte, single_pred) else 'single'

# Non-zero performance
nz = yte > 0
if nz.sum() > 10:
    print(f"\n  On MORTALITY DAYS only ({nz.sum()} days):")
    print(f"    MAE:  {mean_absolute_error(yte[nz], best_pred[nz]):.4f}")
    print(f"    R²:   {r2_score(yte[nz], best_pred[nz]):.4f}")

print(f"\n  ✅ Best: {best_name} | Time: {duration:.1f}s")


# ═══════════════════════════════════════════════════════════════
# CELL 7: Feature Importance + JAPFA Breach Impact
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("🔍 WHAT DRIVES MORTALITY?")
print("=" * 60)

importances = single.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\nTop 25 factors:\n")
for rank, (name, imp) in enumerate(feat_imp[:25], 1):
    bar = '█' * int(imp / max(importances) * 40)
    print(f"  {rank:2d}. {name:35s} {imp:6.0f} {bar}")

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
top = 25
names = [x[0] for x in feat_imp[:top]][::-1]
vals = [x[1] for x in feat_imp[:top]][::-1]
cmap = {'breach':'#e74c3c','temp':'#f39c12','heat':'#f39c12','cold':'#3498db',
        'humid':'#3498db','nh3':'#2ecc71','weight':'#9b59b6','doc':'#9b59b6',
        'water':'#1abc9c','mort':'#e67e22','flock':'#e67e22','age':'#e67e22'}
colors = []
for n in names:
    c = '#95a5a6'
    for k, v in cmap.items():
        if k in n.lower():
            c = v; break
    colors.append(c)
ax.barh(names, vals, color=colors, alpha=0.85)
ax.set_title('Top 25 Mortality Drivers (incl. JAPFA Breaches)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_v3.png', dpi=150)
plt.show()

# Breach impact
print(f"\n🚨 JAPFA BREACH IMPACT ON MORTALITY:")
for breach_col in ['temp_breach','humidity_breach','nh3_breach','total_breach_score']:
    if breach_col in daily.columns:
        no_breach = daily[daily[breach_col] == 0]['daily_mortality'].mean()
        breach = daily[daily[breach_col] > 0]['daily_mortality'].mean()
        increase = ((breach / max(no_breach, 0.001)) - 1) * 100
        print(f"  {breach_col:25s}: No breach={no_breach:.3f}, Breach={breach:.3f} → {increase:+.0f}% increase")


# ═══════════════════════════════════════════════════════════════
# CELL 8: Weight Impact Analysis
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("⚖️ WEIGHT IMPACT ANALYSIS")
print("=" * 60)

w_data = daily[daily['avg_weight'] > 0].copy()
if len(w_data) > 100:
    # Correlation
    w_corr = w_data[['avg_weight','weight_vs_standard','weight_gain_vs_doc',
                      'weight_to_age_ratio','daily_mortality','avg_temp','avg_humidity','avg_nh3']].corr()

    print("\n  Weight correlations with mortality:")
    for col in ['avg_weight','weight_vs_standard','weight_gain_vs_doc','weight_to_age_ratio']:
        c = w_corr.loc[col, 'daily_mortality']
        print(f"    {col:30s}: {c:+.4f}")

    # Weight by flock age
    print("\n  Weight progression by flock age (week):")
    weekly_wt = w_data.groupby('flock_age_weeks').agg(
        avg_wt=('avg_weight','mean'), avg_mort=('daily_mortality','mean')
    ).head(7)
    for _, r in weekly_wt.iterrows():
        print(f"    Week {int(r.name):2d}: Weight={r['avg_wt']:.0f}g, Mortality={r['avg_mort']:.3f}")

    # Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(w_data['avg_weight'], w_data['daily_mortality'], alpha=0.1, s=5, c='#e74c3c')
    ax1.set_xlabel('Weight (g)'); ax1.set_ylabel('Daily Mortality')
    ax1.set_title('Weight vs Mortality')

    wk = w_data.groupby('flock_age_weeks').agg(w=('avg_weight','mean'),m=('daily_mortality','mean')).reset_index()
    ax2.bar(wk['flock_age_weeks'], wk['m'], color='#e74c3c', alpha=0.7, label='Mortality')
    ax2t = ax2.twinx()
    ax2t.plot(wk['flock_age_weeks'], wk['w'], 'b-o', label='Weight')
    ax2.set_xlabel('Week'); ax2.set_ylabel('Mortality', color='red')
    ax2t.set_ylabel('Weight (g)', color='blue')
    ax2.set_title('Weekly Weight vs Mortality')
    plt.tight_layout()
    plt.savefig('weight_impact.png', dpi=150)
    plt.show()
else:
    print("  ⚠️ Not enough weight data for analysis")


# ═══════════════════════════════════════════════════════════════
# CELL 9: Actionable Recommendations
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("📋 ACTIONABLE RECOMMENDATIONS")
print("=" * 60)

# Compute thresholds from data
p90 = daily['daily_mortality'].quantile(0.90)
high = daily[daily['daily_mortality'] >= p90]
low = daily[daily['daily_mortality'] <= daily['daily_mortality'].quantile(0.25)]

def get_alarm(col):
    h = high[col].mean(); l = low[col].mean()
    return round(l + (h-l)*0.6, 1)

print(f"""
╔══════════════════════════════════════════════════════════╗
║            RECOMMENDED ACTIONS BY PARAMETER              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  🌡️ TEMPERATURE                                         ║
║  ├── JAPFA Standard: Day-specific (starts 33°C, drops)  ║
║  ├── ALARM: > {get_alarm('avg_temp')}°C above standard OR > 3°C swing    ║
║  ├── ACTION: Activate pad cooling + increase fans       ║
║  └── Monitor: temp_breach field from model              ║
║                                                          ║
║  💧 HUMIDITY                                             ║
║  ├── JAPFA Standard: 60-70%                             ║
║  ├── ALARM: > {get_alarm('avg_humidity')}% or < 45%                         ║
║  ├── ACTION: Adjust ventilation rate + check foggers    ║
║  └── High humidity + high temp = deadly combination     ║
║                                                          ║
║  🧪 AMMONIA (NH3)                                        ║
║  ├── JAPFA Standard: < 10 ppm (Good), 16-18 (Poor)     ║
║  ├── ALARM: > {get_alarm('avg_nh3')} ppm                                 ║
║  ├── ACTION: Increase ventilation immediately           ║
║  ├── ACTION: Check litter moisture + water leaks        ║
║  └── NH3 > 20 ppm = BAD category, emergency action     ║
║                                                          ║
║  💧 WATER CONSUMPTION                                    ║
║  ├── ALARM: Sudden drop > 20% from previous day        ║
║  ├── ACTION: Check water lines, pressure, nipples       ║
║  └── Low water = early disease indicator                ║
║                                                          ║
║  ⚖️ WEIGHT                                               ║
║  ├── Compare daily vs JAPFA weight standard             ║
║  ├── ALARM: > 10% below standard for flock age          ║
║  ├── ACTION: Review feed quality + intake               ║
║  └── Underweight birds are more vulnerable              ║
║                                                          ║
║  🐣 DOC (Day Old Chick)                                  ║
║  ├── Optimal: 40-45g                                    ║
║  ├── ALARM: < 38g (underweight chicks)                  ║
║  └── Low DOC → extra care in first 14 days              ║
║                                                          ║
║  🔥 COMBINED RISK                                        ║
║  ├── High temp + High humidity + High NH3 = EMERGENCY   ║
║  ├── Multiple JAPFA breaches same day = HIGH ALERT      ║
║  └── 3+ consecutive breach days = CRITICAL              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════════
# CELL 10: Actual vs Predicted Charts
# ═══════════════════════════════════════════════════════════════

test_dates = daily['Date'].values[split:]

fig, axes = plt.subplots(3, 1, figsize=(18, 14))
show = min(90, len(yte))
axes[0].plot(range(show), yte[-show:], 'r-', alpha=0.8, label='Actual', linewidth=1.5)
axes[0].plot(range(show), best_pred[-show:], 'b--', alpha=0.8, label='Predicted', linewidth=1.5)
axes[0].set_title('Actual vs Predicted — Last 90 Days', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(yte, best_pred, alpha=0.2, s=8, c='#3498db')
mx = max(yte.max(), best_pred.max())
axes[1].plot([0, mx], [0, mx], 'r--')
axes[1].set_title('Scatter: Actual vs Predicted', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
axes[1].grid(True, alpha=0.3)

err = best_pred - yte
axes[2].hist(err, bins=50, color='#2ecc71', alpha=0.7)
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_title(f'Error Distribution (mean={err.mean():.3f}, std={err.std():.3f})', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_accuracy_v3.png', dpi=150)
plt.show()


# ═══════════════════════════════════════════════════════════════
# CELL 11: Save Model to Drive
# ═══════════════════════════════════════════════════════════════

import joblib
from datetime import datetime

alerts_thresholds = {}
for col in ['avg_temp','avg_humidity','avg_nh3','heat_stress','water_consumption']:
    if col in daily.columns:
        alerts_thresholds[col] = get_alarm(col)

model_artifact = {
    'classifier': clf,
    'regressor': reg,
    'single_model': single,
    'best_method': best_name,
    'scaler': scaler,
    'farm_encoder': farm_enc,
    'shed_encoder': shed_enc,
    'location_encoder': loc_enc,
    'feature_names': feature_cols,
    'alerts': alerts_thresholds,
    'japfa_standards': {
        'temp': temp_std.to_dict('records') if temp_std is not None else [],
        'humidity': hum_std.to_dict('records') if hum_std is not None else [],
        'nh3': nh3_std.to_dict('records') if nh3_std is not None else [],
        'weight': wt_std.to_dict('records') if wt_std is not None else [],
        'mortality': mort_std.to_dict('records') if mort_std is not None else [],
    },
    'metrics': {
        'clf_f1': round(float(f1_score(yte_bin, clf_pred)), 4),
        'clf_accuracy': round(float(accuracy_score(yte_bin, clf_pred)), 4),
        'clf_recall': round(float(recall_score(yte_bin, clf_pred)), 4),
        'mae': round(float(mean_absolute_error(yte, best_pred)), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(yte, best_pred))), 4),
        'r2': round(float(r2_score(yte, best_pred)), 4),
        'top_features': [{'name':n,'importance':int(v)} for n,v in feat_imp[:15]],
        'farms': list(farm_enc.classes_),
        'sheds': list(shed_enc.classes_),
        'date_range': f"{daily['Date'].min().date()} to {daily['Date'].max().date()}",
    },
    'training_date': datetime.now().isoformat(),
    'n_samples': len(daily),
}

joblib.dump(model_artifact, 'poultry_model_v3.pkl')
size = os.path.getsize('poultry_model_v3.pkl') / 1024 / 1024
joblib.dump(model_artifact, '/content/drive/MyDrive/poultry_model_v3.pkl')

for f in ['feature_importance_v3.png','weight_impact.png','prediction_accuracy_v3.png']:
    if os.path.exists(f):
        import shutil; shutil.copy(f, f'/content/drive/MyDrive/{f}')

print(f"\n✅ Model v3 saved! ({size:.1f} MB)")
print(f"   MAE:  {model_artifact['metrics']['mae']:.4f}")
print(f"   R²:   {model_artifact['metrics']['r2']:.4f}")
print(f"   F1:   {model_artifact['metrics']['clf_f1']:.4f}")

from google.colab import files
files.download('poultry_model_v3.pkl')
print("\n🎉 Download started! Deploy with FastAPI.")
