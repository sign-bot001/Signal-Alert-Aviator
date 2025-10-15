# app.py
"""
Signal Alert AVTR - Version améliorée
- Intègre un petit historique fourni par l'utilisateur (format décimal avec virgule accepté).
- Feature engineering riche (lags, rolling stats, momentum, pct change, time features).
- Ensemble de modèles (RandomForest, ExtraTrees, GradientBoosting) + stacking.
- Estimation d'incertitude via dispersion (arbres) + ensemble dispersion.
- Backtest simple pour simuler une stratégie de cash-out.
- Validation temporelle (TimeSeriesSplit).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import io
import matplotlib.pyplot as plt

# sklearn
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Signal Alert AVTR (amélioré)", layout="wide")

# --- CSS dark hacker style ---
st.markdown(
    """
    <style>
    .reportview-container, .main, header, footer {background-color: #050608;}
    .stApp { color: #cfeef8; }
    .title {font-family: 'Courier New', Courier, monospace; color: #7afcff; font-size:26px;}
    .subtitle {color:#9ff0ff; font-size:13px;}
    .stButton>button {background-color:#0b6cff;color:white;border-radius:8px;}
    .big {font-size:18px; color:#bfffaa;}
    .small {font-size:12px; color:#9fd;}
    .pred {font-weight:bold; color:#7aff7a;}
    .conf {font-weight:bold; color:#f6ff7a;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">⚡ Signal Alert AVTR — Advanced</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ensemble ML · Features avancés · Backtest · Estimation d\'incertitude</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar params
st.sidebar.header("Paramètres")
tz_choice = "Asia/Seoul"
use_sample_btn = st.sidebar.button("Charger l'historique fourni (sample)")
history_file = st.sidebar.file_uploader("Ou charge un CSV (timestamp,multiplier) :", type=["csv"])
min_rounds = st.sidebar.number_input("Min historique (minutes)", value=120, step=60)
n_estimators = st.sidebar.slider("Arbres (RandomForest/ExtraTrees)", 50, 400, 150)
max_lags = st.sidebar.slider("Nombre de lags (minutes)", 5, 60, 20)
predict_minutes = st.sidebar.number_input("Minutes à prédire", value=60, min_value=1, max_value=240, step=1)
conf_threshold = st.sidebar.slider("Seuil confiance pour alertes (%)", 30, 100, 50)
cashout_threshold = st.sidebar.number_input("Seuil cote pour cash-out (x)", value=2.0, step=0.1)
st.sidebar.markdown("**Note**: Les timestamps sont convertis en KST (Asia/Seoul).")

# ---------- Sample history (user provided) ----------
SAMPLE_RAW = """
1,3
1,23
1,56
2,25
1,15
13,09
20,91
2,05
10,17
3,82
1
1,46
1,4
1,73
1,17
1,00
26,60
8,6
1,27
1,46
1,36
1,76
3,61
2,74
1,47
3,7
1,05
"""

def parse_sample_to_df(sample_text):
    # each line is a multiplier; we will create timestamps at 1-minute intervals ending now (KST)
    lines = [ln.strip() for ln in sample_text.strip().splitlines() if ln.strip()]
    # convert commas to dots & parse floats
    vals = []
    for ln in lines:
        ln2 = ln.replace(',', '.')
        try:
            vals.append(float(ln2))
        except:
            # try stripping non-numeric
            filtered = ''.join(ch for ch in ln2 if ch.isdigit() or ch=='.' or ch=='-')
            if filtered:
                vals.append(float(filtered))
    # create timestamps: last timestamp = now in KST, then go back per minute
    tz = pytz.timezone(tz_choice)
    last_ts = datetime.now(tz)
    timestamps = [last_ts - timedelta(minutes=(len(vals)-1-i)) for i in range(len(vals))]
    df = pd.DataFrame({'timestamp': timestamps, 'multiplier': vals})
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.tz_localize(tz)
    return df

# ---------- Data loading & preprocessing ----------
def load_csv_handle_decimal(file_obj):
    # read as text, convert comma decimals if necessary
    txt = file_obj.getvalue().decode('utf-8') if hasattr(file_obj, 'getvalue') else file_obj.read()
    # replace semicolons with commas if user used ; as delimiter
    # convert decimal commas to decimal points where appropriate (but be cautious)
    # A safe approach: if lines contain only single numeric column with comma as decimal sep,
    # replace comma with dot where there's no other comma delimiting fields.
    lines = txt.splitlines()
    processed_lines = []
    for line in lines:
        # If the line has exactly one comma and no other separators, it's likely decimal comma
        if line.count(',') == 1 and (line.count(';')==0) and (line.count(',') < 3):
            # ambiguous: could be two columns; we handle common CSV with header too
            parts = line.split(',')
            # if both parts are numeric-like with decimals, then assume decimal comma in first part?
            # Simpler: replace comma by dot if the line has no other separators and looks like single number
            # We'll attempt to infer: if there is only one field when splitting by comma after checking for timestamp pattern, keep safe
            processed_lines.append(line)
        else:
            processed_lines.append(line)
    # Try reading with pandas, letting it infer separators; to handle decimal commas, try both.
    try:
        from io import StringIO
        s = txt.replace(';', ',')  # common semicolon-to-comma
        # replace decimal commas in numeric-only files (heuristic): if file has one column per line and commas inside numbers, replace commas with dots
        # detect if majority of lines match numeric with comma pattern (e.g., "1,23")
        numeric_comma_count = sum(1 for l in s.splitlines() if l.strip() and all(ch.isdigit() or ch in ",.- " for ch in l.strip()))
        if numeric_comma_count / max(1, len(s.splitlines())) > 0.5:
            # treat as one-column numeric list with comma decimals
            s2 = s.replace(',', '.')
            df = pd.read_csv(StringIO(s2), header=None, names=['multiplier'])
            # create timestamps as consecutive minutes ending now
            tz = pytz.timezone(tz_choice)
            last_ts = datetime.now(tz)
            timestamps = [last_ts - timedelta(minutes=(len(df)-1-i)) for i in range(len(df))]
            df['timestamp'] = timestamps
            df = df[['timestamp','multiplier']]
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(tz)
            return df
        # else try normal CSV read
        df = pd.read_csv(StringIO(s))
        return df
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}")
        return None

def prepare_df(df_raw):
    # Accepts dataframes with either 'multiplier' or single-column numeric, or 'timestamp'+'multiplier'
    df = df_raw.copy()
    # Normalize column names
    cols = [c.lower() for c in df.columns]
    if 'multiplier' not in cols and 'mult' not in ''.join(cols) and df.shape[1]==1:
        # single column of multipliers
        df = df.rename(columns={df.columns[0]:'multiplier'})
        # create timestamps: last = now KST
        tz = pytz.timezone(tz_choice)
        last_ts = datetime.now(tz)
        timestamps = [last_ts - timedelta(minutes=(len(df)-1-i)) for i in range(len(df))]
        df['timestamp'] = timestamps
        df = df[['timestamp','multiplier']]
    else:
        # try to locate timestamp and multiplier columns
        ts_col = None
        mult_col = None
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ['time','date','timestamp']):
                ts_col = c
            if any(k in lc for k in ['mult','cote','value','rate']):
                mult_col = c
        if ts_col is None:
            # assume first column is timestamp
            ts_col = df.columns[0]
        if mult_col is None:
            # pick second column if exists, else first
            if len(df.columns) >= 2:
                mult_col = df.columns[1]
            else:
                mult_col = df.columns[0]
        df = df[[ts_col, mult_col]]
        df.columns = ['timestamp','multiplier']
        # clean multiplier: replace comma decimal
        df['multiplier'] = df['multiplier'].astype(str).str.strip().str.replace(',', '.')
        df['multiplier'] = pd.to_numeric(df['multiplier'], errors='coerce')
        # parse timestamps robustly
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        # convert to timezone KST
        try:
            df['timestamp'] = df['timestamp'].dt.tz_convert(pytz.timezone(tz_choice))
        except Exception:
            df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC).dt.tz_convert(pytz.timezone(tz_choice))
        # resample to 1-minute bins (last value)
        df = df.set_index('timestamp').resample('1T').agg({'multiplier':'last'}).ffill().reset_index()
    # final cleaning: drop NaNs
    df = df.dropna(subset=['multiplier'])
    # ensure tz-aware timestamps
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.timezone(tz_choice))
    return df

# ---------- Feature engineering ----------
def add_time_features(df):
    # add minute_of_hour, hour_of_day, dayofweek
    df['minute'] = df['timestamp'].dt.minute
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def create_lag_features(df, lags=20):
    series = df['multiplier']
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = series.shift(lag)
    # rolling features
    df['roll_mean_5'] = series.rolling(window=5, min_periods=1).mean().shift(1)
    df['roll_std_5']  = series.rolling(window=5, min_periods=1).std().shift(1).fillna(0)
    df['roll_mean_15'] = series.rolling(window=15, min_periods=1).mean().shift(1)
    df['roll_std_15']  = series.rolling(window=15, min_periods=1).std().shift(1).fillna(0)
    df['momentum_3'] = series / series.shift(3) - 1
    df['pct_change_1'] = series.pct_change(1).shift(1).fillna(0)
    df['pct_change_3'] = series.pct_change(3).shift(1).fillna(0)
    # rolling volatility normalized
    df['volatility_15'] = df['roll_std_15'] / (df['roll_mean_15']+1e-9)
    return df

# ---------- Modeling helpers ----------
def build_models(n_estimators):
    models = {
        'rf': RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42),
        'et': ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=43),
        'gbr': GradientBoostingRegressor(n_estimators=int(n_estimators/2), random_state=44)
    }
    return models

def fit_stack(X_train, y_train, base_models, meta_model=LinearRegression()):
    # simple stacking: train base models on full train set, generate out-of-fold predictions via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    meta_features = np.zeros((X_train.shape[0], len(base_models)))
    # out-of-fold
    for i, (name, model) in enumerate(base_models.items()):
        oof = np.zeros(X_train.shape[0])
        for train_idx, val_idx in tscv.split(X_train):
            m = clone(model)
            m.fit(X_train[train_idx], y_train[train_idx])
            oof[val_idx] = m.predict(X_train[val_idx])
        meta_features[:, i] = oof
        # finally fit model on full train
        base_models[name].fit(X_train, y_train)
    # fit meta
    meta_model.fit(meta_features, y_train)
    return base_models, meta_model

def predict_with_uncertainty(X_input, base_models, meta_model):
    # base model predictions
    preds = []
    for name, m in base_models.items():
        preds.append(m.predict(X_input))
    preds = np.vstack(preds)  # shape (n_models, n_samples)
    # meta input
    meta_in = preds.T
    final_pred = meta_model.predict(meta_in)
    # uncertainty estimate: std across base models + relative disagreement
    std_across_models = np.std(preds, axis=0)
    # normalized confidence (map std -> 0..100)
    conf = std_to_confidence(std_across_models)
    return final_pred, std_across_models, conf

def std_to_confidence(std_array):
    if np.all(std_array == 0):
        return np.ones_like(std_array)*100.0
    # map using percentile
    clip_val = np.percentile(std_array, 95) + 1e-9
    conf = 100 * (1 - (std_array / clip_val))
    conf = np.clip(conf, 0, 100)
    return conf

# ---------- Backtest (simple) ----------
def simple_backtest(df_history, preds_df, stake=1.0, cashout_threshold=2.0, conf_threshold=50.0):
    """
    Strategy: for each predicted minute, if predicted_multiplier >= cashout_threshold AND confidence >= conf_threshold:
      - assume we cash out at actual multiplier in history (we have historical true next-minute multipliers only for in-sample)
      - Profit = stake * (actual_multiplier - 1)
    This is a simplistic backtest on historical overlap where actual future is known.
    """
    # join by timestamp for pred/actual (assuming preds_df has timestamp_kst)
    merged = preds_df.merge(df_history[['timestamp','multiplier']], left_on='timestamp_kst', right_on='timestamp', how='left', suffixes=('','_actual'))
    profits = []
    trades = 0
    wins = 0
    for _, row in merged.iterrows():
        pred = row['predicted_multiplier']
        conf = row['confidence_0_100']
        actual = row['multiplier_actual'] if 'multiplier_actual' in row else np.nan
        if (not pd.isna(actual)) and (pred >= cashout_threshold) and (conf >= conf_threshold):
            trades += 1
            profit = stake * (actual - 1)
            profits.append(profit)
            if profit > 0:
                wins += 1
    total = sum(profits)
    roi = (total / (stake*max(1, trades))) * 100 if trades>0 else 0.0
    win_rate = (wins / trades * 100) if trades>0 else 0.0
    return {'trades':trades, 'total_profit':total, 'roi_percent_per_trade':roi, 'win_rate_percent':win_rate}

# ---------- Main UI flow ----------
data_loaded = None
if use_sample_btn:
    df_raw = parse_sample_to_df(SAMPLE_RAW)
    df = prepare_df(df_raw)
    data_loaded = df
    st.success("Historique sample chargé.")
elif history_file is not None:
    df_try = load_csv_handle_decimal(history_file)
    if df_try is None:
        st.error("Impossible de parser le CSV.")
        st.stop()
    df = prepare_df(df_try)
    data_loaded = df

if data_loaded is None:
    st.info("Charge un CSV d'historique ou clique sur 'Charger l'historique fourni (sample)'.\nLe CSV attendu: colonnes timestamp,multiplier OU une simple liste de multipliers (une par ligne).")
    st.stop()

df = data_loaded.copy()
st.markdown("### Aperçu de l'historique (converti en KST, échantillonné par minute)")
st.dataframe(df.tail(20))

if len(df) < max_lags + 10:
    st.warning(f"Historique court ({len(df)} lignes). Les performances peuvent être instables. Recommande >= {max_lags+50} minutes.")

# feature engineering
df = add_time_features(df)
df = create_lag_features(df, lags=max_lags)
# drop rows with NaN after shifting
df = df.dropna().reset_index(drop=True)
st.markdown(f"Features créées — shape final: {df.shape}")

# Prepare X/y
feature_cols = [c for c in df.columns if c not in ['timestamp','multiplier']]
X = df[feature_cols].values
y = df['multiplier'].values

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split using time order (80/20)
split_idx = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
ts_train_df = df.iloc[:split_idx].copy()
ts_test_df = df.iloc[split_idx:].copy()

st.markdown("### Entrainement des modèles (ensemble + stacking)")
models = build_models(n_estimators=n_estimators)
base_models, meta = fit_stack(X_train, y_train, models, meta_model=Ridge(alpha=1.0))

# Evaluate on test
y_pred_test, std_models_test, conf_test = predict_with_uncertainty(X_test, base_models, meta)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)
st.markdown(f"- MAE (test): {mae:.4f}")
st.markdown(f"- RMSE (test): {rmse:.4f}")

# iterative forecast for next N minutes using last window
def iterative_predict_next(df_full, max_lags, steps, scaler, base_models, meta):
    # Start from last row in df (which has features up-to-date)
    last_features = df_full[feature_cols].iloc[-1:].values  # shape (1, n_features)
    window_df = df_full.copy()
    preds = []
    stds = []
    confs = []
    last_ts = window_df['timestamp'].iloc[-1]
    for i in range(steps):
        x_scaled = scaler.transform(last_features)
        pred, std_model, conf = predict_with_uncertainty(x_scaled, base_models, meta)
        p = float(pred[0])
        s = float(std_model[0])
        c = float(conf[0])
        preds.append(p); stds.append(s); confs.append(c)
        # append pseudo-row with predicted multiplier to window_df to build next features
        next_ts = last_ts + timedelta(minutes=1)
        next_row = {'timestamp': next_ts, 'multiplier': p, 'minute': next_ts.minute, 'hour': next_ts.hour, 'dayofweek': next_ts.dayofweek}
        # create lag features: take last max_lags multipliers then append p
        last_mult_series = list(window_df['multiplier'].values)
        last_mult_series.append(p)
        # reconstruct feature row
        for lag in range(1, max_lags+1):
            # lag_1 is previous minute -> last_mult_series[-1 - (lag-1) - 1] = -lag-? simpler:
            idx = len(last_mult_series) - 1 - lag
            val = last_mult_series[idx] if idx >= 0 else np.nan
            next_row[f'lag_{lag}'] = val
        # rolling features
        ser = pd.Series(last_mult_series)
        next_row['roll_mean_5'] = ser.shift(1).rolling(window=5, min_periods=1).mean().iloc[-1]
        next_row['roll_std_5'] = ser.shift(1).rolling(window=5, min_periods=1).std().iloc[-1] if ser.shift(1).rolling(window=5, min_periods=1).std().iloc[-1] == ser.shift(1).rolling(window=5, min_periods=1).std().iloc[-1] else 0.0
        next_row['roll_mean_15'] = ser.shift(1).rolling(window=15, min_periods=1).mean().iloc[-1]
        next_row['roll_std_15'] = ser.shift(1).rolling(window=15, min_periods=1).std().iloc[-1] if ser.shift(1).rolling(window=15, min_periods=1).std().iloc[-1] == ser.shift(1).rolling(window=15, min_periods=1).std().iloc[-1] else 0.0
        next_row['momentum_3'] = ser.iloc[-1] / ser.shift(3).iloc[-1] - 1 if len(ser) > 3 and ser.shift(3).iloc[-1] != 0 else 0.0
        next_row['pct_change_1'] = ser.pct_change(1).shift(1).iloc[-1] if len(ser)>1 else 0.0
        next_row['pct_change_3'] = ser.pct_change(3).shift(1).iloc[-1] if len(ser)>3 else 0.0
        next_row['volatility_15'] = next_row['roll_std_15'] / (next_row['roll_mean_15'] + 1e-9)
        # ensure ordering of feature_cols
        next_df_row = pd.DataFrame([next_row])
        # fill missing cols if any
        for c in feature_cols:
            if c not in next_df_row.columns:
                next_df_row[c] = 0.0
        next_df_row = next_df_row[feature_cols]
        # set last_features for next iteration
        last_features = next_df_row.values
        window_df = pd.concat([window_df, pd.DataFrame({'timestamp':[next_ts], 'multiplier':[p]})], ignore_index=True)
        last_ts = next_ts
    # build preds DataFrame with timestamps
    pred_times = [df_full['timestamp'].iloc[-1] + timedelta(minutes=i+1) for i in range(len(preds))]
    preds_df = pd.DataFrame({
        'timestamp_kst': pred_times,
        'predicted_multiplier': np.round(preds, 6),
        'std_est': np.round(stds, 6),
        'confidence_0_100': np.round(confs, 2)
    })
    return preds_df

preds_df = iterative_predict_next(df, max_lags, int(predict_minutes), scaler, base_models, meta)

st.markdown("### Prédictions (prochaines {} minutes)".format(len(preds_df)))
col1, col2 = st.columns([1,2])
with col1:
    best_idx = preds_df['predicted_multiplier'].idxmax()
    best_row = preds_df.loc[best_idx]
    st.markdown(f"🔺 **Meilleure opportunité** : {best_row['predicted_multiplier']}x à {best_row['timestamp_kst']} (Confiance {best_row['confidence_0_100']}%)")
    st.markdown(f"📉 Moyenne prédite : {preds_df['predicted_multiplier'].mean():.4f}x")
    st.markdown(f"📊 Écart-type moyen (std est) : {preds_df['std_est'].mean():.6f}")
    st.markdown("---")
    st.markdown("**Backtest simple** (sur période connue si applicable)")
    bt = simple_backtest(df, preds_df, stake=1.0, cashout_threshold=cashout_threshold, conf_threshold=conf_threshold)
    st.markdown(f"- Trades simulés : {bt['trades']}")
    st.markdown(f"- Profit total simulé : {bt['total_profit']:.4f}")
    st.markdown(f"- ROI moyen par trade (%) : {bt['roi_percent_per_trade']:.2f}")
    st.markdown(f"- Taux de succès (%) : {bt['win_rate_percent']:.2f}")
with col2:
    st.dataframe(preds_df.style.format({"predicted_multiplier":"{:.4f}", "confidence_0_100":"{:.2f}"}), height=360)

# Plot history + preds
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(df['timestamp'].tail(300).values, df['multiplier'].tail(300).values, label='historique (dernieres 300 min)')
ax.plot(preds_df['timestamp_kst'].values, preds_df['predicted_multiplier'].values, marker='o', linestyle='--', label='predictions')
ax.set_title("Historique et prédictions (KST)")
ax.set_xlabel("Timestamp (KST)")
ax.set_ylabel("Multiplier")
ax.legend()
plt.xticks(rotation=30)
st.pyplot(fig)

# Download preds
buf = io.StringIO()
preds_df.to_csv(buf, index=False)
st.download_button("Télécharger prédictions CSV", buf.getvalue(), file_name="predictions_signal_alert_avtr.csv", mime="text/csv")

# Show model details & theory
st.markdown("---")
st.markdown("## Théorie, formules & algorithmes utilisés (résumé rapide)")
st.markdown("""
- **Feature engineering** : lags (autocorr), moyennes mobiles (rolling mean), volatilité (rolling std), momentum, pour capturer mémoire courte et patterns.
- **Time Series Cross Validation** : *TimeSeriesSplit* pour garder la causalité temporelle (pas de fuite d'information).
- **Ensemble & Stacking** : combiner RandomForest, ExtraTrees, GradientBoosting pour réduire variance/biais. Méta-modèle (Ridge/Linear) pour apprendre la combinaison optimale.
- **Estimation d'incertitude** : dispersion des prédictions des modèles de base (std) -> convertie en score de confiance 0–100.
- **Backtest** : simulation naïve pour estimer ROI basé sur condition predicted >= seuil et confiance >= seuil (simple, non financier).
- **Statistiques** : usage implicite du concept de moyenne mobile (rolling mean), variance (rolling std), et propriétés d'autocorr. Le **Central Limit Theorem (CLT)** justifie que moyennes d’échantillon tendent vers une distribution normale (utile pour l'interprétation d'erreurs), mais **attention** : les rounds de jeux multiplicateurs ne sont pas nécessairement iid.
- **Algorithmes** : RandomForest (bagging d'arbres), ExtraTrees (bagging plus aléatoire), GradientBoosting (boosting), Stacking (combinaison).
""")

st.markdown("### Avertissement légal & d'usage")
st.markdown("""
