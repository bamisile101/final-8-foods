"""
routes/predict.py
-----------------
Prediction endpoints:
  GET /api/predict/<food>/<year>/<month>  → single month prediction
  GET /api/predict/<food>/future          → 12-month forecast
  POST /api/predict/custom                → custom prediction with body params
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, jsonify, request
from model_loader import get_model, get_meta, resolve_food
from predictor import predict_single, predict_future

predict_bp = Blueprint('predict', __name__)

MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August', 'September',
    'October', 'November', 'December'
]


# ── GET /api/predict/<food>/<year>/<month> ─────
@predict_bp.route('/<food>/<int:year>/<int:month>', methods=['GET'])
def predict_one(food, year, month):
    """
    Predict the price of a food item for a specific year and month.

    URL params:
        food  : rice | beans | garri | yam
        year  : e.g. 2026
        month : 1-12

    Example:
        GET /api/predict/rice/2026/8
    """
    # Validate food
    food_key, err = resolve_food(food)
    if err:
        return jsonify({'error': err, 'status': 400}), 400

    # Validate month
    if not (1 <= month <= 12):
        return jsonify({'error': 'Month must be between 1 and 12', 'status': 400}), 400

    # Validate year
    if not (2024 <= year <= 2030):
        return jsonify({'error': 'Year must be between 2024 and 2030', 'status': 400}), 400

    model_bundle = get_model(food_key)
    meta         = get_meta(food_key)
    result       = predict_single(model_bundle, year, month)

    return jsonify({
        'status'      : 200,
        'food'        : meta['label'],
        'food_key'    : food_key,
        'year'        : year,
        'month'       : month,
        'month_name'  : MONTH_NAMES[month],
        'currency'    : 'NGN',
        'unit'        : 'per kg',
        'predictions' : {
            'xgboost'      : result['xgboost'],
            'random_forest': result['random_forest'],
            'average'      : result['average'],
        },
        'last_known_price': meta['last_price'],
        'last_known_date' : meta['last_date'],
        'note': 'Predictions are based on historical patterns from 2007-2026 World Bank data.'
    })


# ── GET /api/predict/<food>/future ────────────
@predict_bp.route('/<food>/future', methods=['GET'])
def predict_future_route(food):
    """
    Get 12-month price forecast for a food item.

    URL params:
        food : rice | beans | garri | yam

    Query params (optional):
        months : number of months ahead (default 12, max 24)

    Example:
        GET /api/predict/rice/future
        GET /api/predict/beans/future?months=6
    """
    food_key, err = resolve_food(food)
    if err:
        return jsonify({'error': err, 'status': 400}), 400

    months = request.args.get('months', 12, type=int)
    months = min(max(months, 1), 24)  # clamp between 1 and 24

    model_bundle = get_model(food_key)
    meta         = get_meta(food_key)
    forecast     = predict_future(model_bundle, months_ahead=months)

    return jsonify({
        'status'     : 200,
        'food'       : meta['label'],
        'food_key'   : food_key,
        'currency'   : 'NGN',
        'unit'       : 'per kg',
        'months_ahead': months,
        'last_known_price': meta['last_price'],
        'last_known_date' : meta['last_date'],
        'forecast'   : forecast,
        'note': 'Forecasts use rolling predictions from trained XGBoost and Random Forest models.'
    })


# ── POST /api/predict/custom ──────────────────
@predict_bp.route('/custom', methods=['POST'])
def predict_custom():
    """
    Predict prices for multiple foods at once.

    Request body (JSON):
    {
        "foods" : ["rice", "beans"],
        "year"  : 2027,
        "month" : 3
    }

    Example:
        POST /api/predict/custom
        Body: {"foods": ["rice", "garri"], "year": 2026, "month": 9}
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body must be JSON', 'status': 400}), 400

    foods_req = data.get('foods', [])
    year      = data.get('year')
    month     = data.get('month')

    # Validate
    if not foods_req:
        return jsonify({'error': 'Provide at least one food in "foods" list', 'status': 400}), 400
    if not year or not month:
        return jsonify({'error': 'Provide "year" and "month"', 'status': 400}), 400
    if not (1 <= month <= 12):
        return jsonify({'error': 'Month must be 1-12', 'status': 400}), 400
    if not (2024 <= year <= 2030):
        return jsonify({'error': 'Year must be 2024-2030', 'status': 400}), 400

    results = {}
    errors  = []

    for food_name in foods_req:
        food_key, err = resolve_food(food_name)
        if err:
            errors.append(err)
            continue
        model_bundle = get_model(food_key)
        meta         = get_meta(food_key)
        pred         = predict_single(model_bundle, year, month)
        results[food_name] = {
            'label'        : meta['label'],
            'xgboost'      : pred['xgboost'],
            'random_forest': pred['random_forest'],
            'average'      : pred['average'],
            'currency'     : 'NGN',
            'unit'         : 'per kg',
        }

    return jsonify({
        'status'    : 200,
        'year'      : year,
        'month'     : month,
        'month_name': MONTH_NAMES[month],
        'predictions': results,
        'errors'    : errors if errors else None,
    })
