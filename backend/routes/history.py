"""
routes/history.py
-----------------
Historical price data endpoints:
  GET /api/history/<food>        → full price history
  GET /api/history/<food>/recent → last N months
  GET /api/history/all           → all 4 foods summary
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, jsonify, request
from model_loader import get_meta, resolve_food, VALID_FOODS

history_bp = Blueprint('history', __name__)


# ── GET /api/history/<food> ────────────────────
@history_bp.route('/<food>', methods=['GET'])
def get_history(food):
    """
    Get full price history for a food item.

    URL params:
        food : rice | beans | garri | yam

    Query params (optional):
        limit : max number of records to return (default: all)
        start : start date filter e.g. 2020-01-01
        end   : end date filter e.g. 2026-01-01

    Example:
        GET /api/history/rice
        GET /api/history/garri?limit=24
        GET /api/history/yam?start=2020-01-01
    """
    food_key, err = resolve_food(food)
    if err:
        return jsonify({'error': err, 'status': 400}), 400

    meta   = get_meta(food_key)
    dates  = meta['history_dates']
    prices = meta['history_prices']

    # Apply date filters
    start_filter = request.args.get('start')
    end_filter   = request.args.get('end')

    records = []
    for d, p in zip(dates, prices):
        if start_filter and d < start_filter:
            continue
        if end_filter and d > end_filter:
            continue
        records.append({'date': d, 'price': p})

    # Apply limit
    limit = request.args.get('limit', type=int)
    if limit:
        records = records[-limit:]  # most recent N

    return jsonify({
        'status'      : 200,
        'food'        : meta['label'],
        'food_key'    : food_key,
        'currency'    : 'NGN',
        'unit'        : 'per kg',
        'total_records': len(records),
        'date_range'  : {
            'start': records[0]['date'] if records else None,
            'end'  : records[-1]['date'] if records else None,
        },
        'data': records
    })


# ── GET /api/history/<food>/recent ────────────
@history_bp.route('/<food>/recent', methods=['GET'])
def get_recent(food):
    """
    Get the most recent price records for a food item.

    Query params:
        months : number of recent months (default 12)

    Example:
        GET /api/history/rice/recent
        GET /api/history/beans/recent?months=6
    """
    food_key, err = resolve_food(food)
    if err:
        return jsonify({'error': err, 'status': 400}), 400

    months = request.args.get('months', 12, type=int)
    months = min(max(months, 1), 120)

    meta   = get_meta(food_key)
    dates  = meta['history_dates'][-months:]
    prices = meta['history_prices'][-months:]

    records = [{'date': d, 'price': p} for d, p in zip(dates, prices)]

    return jsonify({
        'status'  : 200,
        'food'    : meta['label'],
        'food_key': food_key,
        'currency': 'NGN',
        'unit'    : 'per kg',
        'months'  : months,
        'data'    : records
    })


# ── GET /api/history/all ───────────────────────
@history_bp.route('/all', methods=['GET'])
def get_all_summary():
    """
    Get a summary of all 4 food items — latest price, date range, record count.

    Example:
        GET /api/history/all
    """
    summary = {}
    for short_name, food_key in VALID_FOODS.items():
        meta = get_meta(food_key)
        summary[short_name] = {
            'label'        : meta['label'],
            'food_key'     : food_key,
            'latest_price' : meta['last_price'],
            'latest_date'  : meta['last_date'],
            'total_records': len(meta['history_dates']),
            'date_range'   : {
                'start': meta['history_dates'][0],
                'end'  : meta['history_dates'][-1],
            }
        }

    return jsonify({
        'status'  : 200,
        'currency': 'NGN',
        'unit'    : 'per kg',
        'foods'   : summary
    })
