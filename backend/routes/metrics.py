"""
routes/metrics.py
-----------------
Model evaluation metrics endpoints:
  GET /api/metrics       → all food metrics
  GET /api/metrics/<food> → one food's metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, jsonify
from model_loader import get_meta, resolve_food, VALID_FOODS

metrics_bp = Blueprint('metrics', __name__)


# ── GET /api/metrics ───────────────────────────
@metrics_bp.route('/', methods=['GET'])
@metrics_bp.route('', methods=['GET'])
def get_all_metrics():
    """
    Return evaluation metrics (MAE, RMSE) for all food models.

    Example:
        GET /api/metrics
    """
    all_metrics = {}

    for short_name, food_key in VALID_FOODS.items():
        meta = get_meta(food_key)
        xgb_wins = meta['xgboost_mae'] <= meta['random_forest_mae']

        all_metrics[short_name] = {
            'label'   : meta['label'],
            'food_key': food_key,
            'xgboost' : {
                'mae'    : meta['xgboost_mae'],
                'rmse'   : meta['xgboost_rmse'],
                'winner' : xgb_wins,
            },
            'random_forest': {
                'mae'    : meta['random_forest_mae'],
                'rmse'   : meta['random_forest_rmse'],
                'winner' : not xgb_wins,
            },
            'best_model'   : 'XGBoost' if xgb_wins else 'Random Forest',
            'n_train'      : meta['n_train'],
            'n_test'       : meta['n_test'],
        }

    return jsonify({
        'status'  : 200,
        'note'    : 'MAE = Mean Absolute Error, RMSE = Root Mean Squared Error. Lower is better.',
        'currency': 'NGN',
        'metrics' : all_metrics,
    })


# ── GET /api/metrics/<food> ────────────────────
@metrics_bp.route('/<food>', methods=['GET'])
def get_food_metrics(food):
    """
    Return evaluation metrics for a single food item.

    URL params:
        food : rice | beans | garri | yam

    Example:
        GET /api/metrics/rice
    """
    food_key, err = resolve_food(food)
    if err:
        return jsonify({'error': err, 'status': 400}), 400

    meta     = get_meta(food_key)
    xgb_wins = meta['xgboost_mae'] <= meta['random_forest_mae']

    return jsonify({
        'status'    : 200,
        'food'      : meta['label'],
        'food_key'  : food_key,
        'currency'  : 'NGN',
        'note'      : 'Lower MAE and RMSE = better prediction accuracy.',
        'best_model': 'XGBoost' if xgb_wins else 'Random Forest',
        'n_train'   : meta['n_train'],
        'n_test'    : meta['n_test'],
        'xgboost': {
            'model_type': 'Gradient Boosting (XGBoost equivalent)',
            'mae'       : meta['xgboost_mae'],
            'rmse'      : meta['xgboost_rmse'],
            'winner'    : xgb_wins,
        },
        'random_forest': {
            'model_type': 'Random Forest (Ensemble)',
            'mae'       : meta['random_forest_mae'],
            'rmse'      : meta['random_forest_rmse'],
            'winner'    : not xgb_wins,
        },
        'last_known_price': meta['last_price'],
        'last_known_date' : meta['last_date'],
    })
