"""
train_models.py
---------------
Run this script to retrain models when you get new data.

Usage:
    python train_models.py
    python train_models.py --data path/to/new_data.csv
"""

import os
import sys
import pickle
import json
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'nigeria_food_prices.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

FOODS = {
    'rice_NGN' : 'Rice',
    'beans_NGN': 'Beans',
    'garri_NGN': 'Garri',
    'yam_NGN'  : 'Yam',
}

FEAT_COLS = [
    'year', 'month',
    'lag1', 'lag2', 'lag3', 'lag6', 'lag12',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3'
]


def make_features(series: pd.Series, years: pd.Series, months: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'price': series, 'year': years, 'month': months})
    df['lag1']           = df['price'].shift(1)
    df['lag2']           = df['price'].shift(2)
    df['lag3']           = df['price'].shift(3)
    df['lag6']           = df['price'].shift(6)
    df['lag12']          = df['price'].shift(12)
    df['rolling_mean_3'] = df['price'].shift(1).rolling(3).mean()
    df['rolling_mean_6'] = df['price'].shift(1).rolling(6).mean()
    df['rolling_std_3']  = df['price'].shift(1).rolling(3).std()
    return df.dropna().reset_index(drop=True)


def train(data_path: str = DATA_PATH):
    print("=" * 55)
    print("  Nigeria Food Price Prediction — Model Training")
    print("=" * 55)
    print(f"\nLoading data from: {data_path}")

    df = pd.read_csv(data_path)
    df['price_date'] = pd.to_datetime(df['price_date'])
    df = df.sort_values('price_date').reset_index(drop=True)

    print(f"Dataset: {len(df)} rows, {df['price_date'].min().date()} → {df['price_date'].max().date()}\n")

    all_models = {}
    model_meta = {}

    for col, label in FOODS.items():
        print(f"Training models for: {label}")

        sub = df[['price_date', 'year', 'month', col]].dropna().reset_index(drop=True)
        print(f"  → {len(sub)} non-null records")

        if len(sub) < 20:
            print(f"  ⚠ Skipping {label} — insufficient data\n")
            continue

        feat = make_features(sub[col], sub['year'], sub['month'])
        X    = feat[FEAT_COLS].values
        y    = feat['price'].values

        split   = int(len(X) * 0.8)
        X_train = X[:split];  y_train = y[:split]
        X_test  = X[split:];  y_test  = y[split:]

        # ── XGBoost (Gradient Boosting) ────────
        gb = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        )
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_mae  = float(mean_absolute_error(y_test, gb_pred))
        gb_rmse = float(np.sqrt(mean_squared_error(y_test, gb_pred)))

        # ── Random Forest ──────────────────────
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mae  = float(mean_absolute_error(y_test, rf_pred))
        rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))

        winner = 'XGBoost' if gb_mae <= rf_mae else 'Random Forest'
        print(f"  XGBoost      → MAE: ₦{gb_mae:,.0f}  RMSE: ₦{gb_rmse:,.0f}")
        print(f"  Random Forest → MAE: ₦{rf_mae:,.0f}  RMSE: ₦{rf_rmse:,.0f}")
        print(f"  Best model   : {winner}\n")

        all_models[col] = {
            'xgboost'      : gb,
            'random_forest': rf,
            'last_values'  : list(sub[col].values[-12:]),
            'last_date'    : str(sub['price_date'].iloc[-1].date()),
            'last_year'    : int(sub['year'].iloc[-1]),
            'last_month'   : int(sub['month'].iloc[-1]),
            'feature_cols' : FEAT_COLS,
        }

        model_meta[col] = {
            'label'             : label,
            'last_price'        : float(sub[col].iloc[-1]),
            'last_date'         : str(sub['price_date'].iloc[-1].date()),
            'xgboost_mae'       : round(gb_mae, 2),
            'xgboost_rmse'      : round(gb_rmse, 2),
            'random_forest_mae' : round(rf_mae, 2),
            'random_forest_rmse': round(rf_rmse, 2),
            'n_train'           : int(split),
            'n_test'            : int(len(y_test)),
            'history_dates'     : [str(d.date()) for d in sub['price_date']],
            'history_prices'    : [round(float(p), 2) for p in sub[col].values],
        }

    # ── Save ───────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    pkl_path  = os.path.join(MODELS_DIR, 'trained_models.pkl')
    meta_path = os.path.join(MODELS_DIR, 'model_meta.json')

    with open(pkl_path, 'wb') as f:
        pickle.dump(all_models, f)

    with open(meta_path, 'w') as f:
        json.dump(model_meta, f, indent=2)

    print("=" * 55)
    print(f"  Models saved → {pkl_path}")
    print(f"  Metadata saved → {meta_path}")
    print("  Training complete! Restart the server to use new models.")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train food price prediction models')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to CSV dataset')
    args = parser.parse_args()
    train(data_path=args.data)
