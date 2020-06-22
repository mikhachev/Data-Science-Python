from flask import json

from app import app


test_data = {
    "ID": 0,
    "LicAge": 468,
    "Gender": "Male",
    "MariStat": "Other",
    "DrivAge": 50,
    "BonusMalus": 56,
    "RiskArea": 0,
    "VehUsage": "Private",
    "SocioCateg": "CSP1"
}


def test_ml_api():
    response = app.test_client().post(
        '/predict',
        data=json.dumps(test_data),
        content_type='application/json',
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
