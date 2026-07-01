"""
predictor.py
------------
Core prediction logic.
Builds lag features and runs XGBoost or Random Forest models.
"""

import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta


def build_feature_vector(year: int, month: int, history: list) -> np.ndarray:
    """
    Build a single feature vector for prediction.

    Features (must match training):
        year, month,
        lag1, lag2, lag3, lag6, lag12,
        rolling_mean_3, rolling_mean_6, rolling_std_3

    Args:
        year    : target year (int)
        month   : target month 1-12 (int)
        history : list of recent prices (most recent last), at least 12 values

    Returns:
        numpy array shape (1, 10)
    """
    h = history  # alias

    lag1  = h[-1]  if len(h) >= 1  else h[0]
    lag2  = h[-2]  if len(h) >= 2  else h[0]
    lag3  = h[-3]  if len(h) >= 3  else h[0]
    lag6  = h[-6]  if len(h) >= 6  else h[0]
    lag12 = h[-12] if len(h) >= 12 else h[0]

    rm3 = float(np.mean(h[-3:])) if len(h) >= 3 else float(np.mean(h))
    rm6 = float(np.mean(h[-6:])) if len(h) >= 6 else float(np.mean(h))
    rs3 = float(np.std(h[-3:]))  if len(h) >= 3 else 0.0

    return np.array([[year, month, lag1, lag2, lag3, lag6, lag12, rm3, rm6, rs3]])


def predict_single(model_bundle: dict, year: int, month: int,
                   model_type: str = 'both') -> dict:
    """
    Predict price for a specific year/month.

    Args:
        model_bundle : dict from model_loader.get_model()
        year         : int
        month        : int (1-12)
        model_type   : 'xgboost' | 'random_forest' | 'both'

    Returns:
        dict with predictions
    """
    history = list(model_bundle['last_values'])
    X = build_feature_vector(year, month, history)

    result = {'year': year, 'month': month}

    if model_type in ('xgboost', 'both'):
        xgb_price = float(model_bundle['xgboost'].predict(X)[0])
        result['xgboost'] = round(xgb_price, 2)

    if model_type in ('random_forest', 'both'):
        rf_price = float(model_bundle['random_forest'].predict(X)[0])
        result['random_forest'] = round(rf_price, 2)

    if model_type == 'both':
        result['average'] = round((result['xgboost'] + result['random_forest']) / 2, 2)

    return result


def predict_future(model_bundle: dict, months_ahead: int = 12) -> list:
    """
    Predict prices for the next N months, rolling predictions forward.

    Args:
        model_bundle : dict from model_loader.get_model()
        months_ahead : number of months to forecast (default 12)

    Returns:
        list of dicts with date, xgboost, random_forest, average
    """
    history_xgb = list(model_bundle['last_values'])  # rolling history for xgboost
    history_rf  = list(model_bundle['last_values'])  # separate rolling for rf

    last_date = date.fromisoformat(model_bundle['last_date'])
    results   = []

    for i in range(1, months_ahead + 1):
        target_date  = last_date + relativedelta(months=i)
        target_year  = target_date.year
        target_month = target_date.month

        # XGBoost prediction
        X_xgb   = build_feature_vector(target_year, target_month, history_xgb)
        xgb_val = float(model_bundle['xgboost'].predict(X_xgb)[0])
        history_xgb.append(xgb_val)

        # Random Forest prediction
        X_rf   = build_feature_vector(target_year, target_month, history_rf)
        rf_val = float(model_bundle['random_forest'].predict(X_rf)[0])
        history_rf.append(rf_val)

        results.append({
            'date'         : str(target_date.replace(day=1)),
            'year'         : target_year,
            'month'        : target_month,
            'month_name'   : target_date.strftime('%B %Y'),
            'xgboost'      : round(xgb_val, 2),
            'random_forest': round(rf_val, 2),
            'average'      : round((xgb_val + rf_val) / 2, 2),
        })

    return results
