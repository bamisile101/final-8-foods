"""
Nigeria Food Price Prediction System
=====================================
Backend: Flask REST API
Author : Bamisile Samuel Temidayo (SEN/20/5095)
School : Federal University of Technology Akure
Project: Machine Learning for Future Food Price Prediction
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from routes.predict import predict_bp
from routes.history import history_bp
from routes.metrics import metrics_bp

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── Register route blueprints ──────────────────
app.register_blueprint(predict_bp,  url_prefix='/api/predict')
app.register_blueprint(history_bp,  url_prefix='/api/history')
app.register_blueprint(metrics_bp,  url_prefix='/api/metrics')


@app.route('/')
def index():
    return jsonify({
        'project': 'Nigeria Food Price Prediction System',
        'author' : 'Bamisile Samuel Temidayo',
        'matric' : 'SEN/20/5095',
        'school' : 'Federal University of Technology Akure',
        'version': '1.0.0',
        'status' : 'running',
        'endpoints': {
            'GET  /api/predict/<food>/<year>/<month>' : 'Predict price for a specific month',
            'GET  /api/predict/<food>/future'         : 'Get 12-month forecast',
            'GET  /api/history/<food>'                : 'Get full price history',
            'GET  /api/metrics'                       : 'Get all model metrics',
            'GET  /api/metrics/<food>'                : 'Get metrics for one food item',
        }
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'status': 404}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'status': 500}), 500


if __name__ == '__main__':
    print("=" * 55)
    print("  Nigeria Food Price Prediction API")
    print("  FUTA FYP — Bamisile Samuel Temidayo")
    print("  Running on http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)
