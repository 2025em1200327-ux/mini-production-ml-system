import pytest
from fastapi.testclient import TestClient

from src.serving.app import app

VALID = {"customer_id": "TEST-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
         "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
         "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
         "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
         "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 470.2}


@pytest.fixture
def payload():
    return dict(VALID)


@pytest.fixture
def client():
    with TestClient(app) as c:      # runs the lifespan, i.e. real artifact loading
        yield c
