import xgboost as xgb
import lightgbm as lgb
from flask import Flask, request, jsonify
from process_data import process_input

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

app = Flask(__name__)

#xgb_freq = xgb.Booster()
#xgb_freq.load_model('models/xgb_freq.model')
lgb_model_freq = lgb.Booster(model_file='models/lgb_model_freq.model')
lgb_model_avgclaim = lgb.Booster(model_file='models/lgb_model_avgclaim.model')
#xgb_avgclaim = xgb.Booster()
#xgb_avgclaim.load_model('models/xgb_avgclaim.model')

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@app.route("/")
def index():
    return "API for predict service"


@app.route("/predict", methods=['POST'])
def predict():
    json_input = request.json

    # Request logging
    current_datetime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datetime} request from {ip_address}: {request.json}')
    start_prediction = time()

    _id = json_input['ID']
    df = process_input(json_input)
    
    prediction_freq = lgb_model_freq.predict(lgb.Dataset(df.values)) - 1
    value_freq = prediction_freq[0]

    prediction_avgclaim = lgb_model_avgclaim.predict(lgb.Dataset(df.values))
    value_avgclaim = prediction_avgclaim[0]

    value_burningcost = value_freq * value_avgclaim

    result = {
        'ID': str(_id),
        'value_freq': str(round(value_freq, 2)),
        'value_avgclaim': str(round(value_avgclaim, 2)),
        'value_burningcost': str(round(value_burningcost, 2))
    }

    # Response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datetime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{current_datetime} predicted for {duration} msec: {result}\n')

    return jsonify(result)


@app.errorhandler(Exception)
def exceptions(e):
    current_datetime = strftime('[%Y-%b-%d %H:%M:%S]')
    error_message = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
                 current_datetime,
                 request.remote_addr,
                 request.method,
                 request.scheme,
                 request.full_path,
                 error_message)
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run('127.0.0.1', port='5000', debug=True)
