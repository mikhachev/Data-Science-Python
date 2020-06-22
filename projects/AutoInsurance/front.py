import json
import requests
import lightgbm as lgb

from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time


from process_data import process_input


def send_json(data):
    url = 'http://127.0.0.1:5000/predict'
    headers = {'content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response


class ClientDataForm(FlaskForm):
    id = IntegerField('ID', validators=[DataRequired()])
    lic_age = IntegerField('Водительский стаж', validators=[DataRequired()])
    gender = SelectField('Пол', choices=[('Male', 'М'), ('Female', 'Ж')])
    mari_stat = SelectField('Семейное положение', choices=[('Alone', 'Alone'), ('Other', 'Other')])
    driv_age = IntegerField('Возраст водителя', validators=[DataRequired()])
    haskmlimit = SelectField('Kм лимит', choices=[('no', 'нет'), ('yes', 'да')])
    bonus_malus = IntegerField('Коэффициент BonusMalus (0.0-1.0)', validators=[DataRequired()])
    risk_area = IntegerField('Зона риска (1-9)', validators=[DataRequired()])
    veh_usage = SelectField('Цель использования т/с', choices=[
        ('VehUsg_Private', 'Личные цели'),
        ('VehUsg_Private+trip to office', 'Личные цели + поездки на работу'),
        ('VehUsg_Professional', 'Для работы'),
        ('VehUsg_Professional run', 'Для участия в заездах')
    ])
    socio_categ = SelectField('Социальная категория', choices=[
        ('CSP1', 'CSP1'),
        ('CSP2', 'CSP2'),
        ('CSP3', 'CSP3'),
        ('CSP4', 'CSP4'),
        ('CSP5', 'CSP5'),
        ('CSP6', 'CSP6'),
        ('CSP7', 'CSP7')
    ])


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

lgb_model_freq = lgb.Booster(model_file='models/lgb_model_freq.model')
lgb_model_avgclaim = lgb.Booster(model_file='models/lgb_model_avgclaim.model')


# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


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

    #print(lgb.Dataset(df.values).data)
    prediction_freq = lgb_model_freq.predict(lgb.Dataset(df.values).data) - 1
    value_freq = prediction_freq[0]

    prediction_avgclaim = lgb_model_avgclaim.predict(lgb.Dataset(df.values).data)
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


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    print(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    if request.method == 'POST':
        data = dict()
        data['ID'] = request.form.get('id')
        data['LicAge'] = float(request.form.get('lic_age'))
        data['Gender'] = request.form.get('gender')
        data['MariStat'] = request.form.get('mari_stat')
        data['DrivAge'] = float(request.form.get('driv_age'))
        data['HasKmLimit'] = request.form.get('haskmimit')

        data['BonusMalus'] = float(request.form.get('bonus_malus'))
        data['RiskArea'] = float(request.form.get('risk_area'))
        data['VehUsage'] = request.form.get('veh_usage')
        data['SocioCateg'] = request.form.get('socio_categ')
        try:
            response = send_json(data)
            response = response.text
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000', debug=True)
