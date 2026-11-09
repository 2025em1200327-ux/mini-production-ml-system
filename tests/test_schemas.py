import math

import pytest
from pydantic import ValidationError

from src.serving.schemas import BatchRequest, CustomerFeatures
from tests.conftest import VALID


def test_valid_payload_accepted():
    assert CustomerFeatures(**VALID).tenure == 5


def test_missing_total_charges_allowed():
    p = dict(VALID); p.pop("TotalCharges")
    assert CustomerFeatures(**p).TotalCharges is None


def test_whitespace_is_stripped():
    assert CustomerFeatures(**{**VALID, "Partner": " Yes "}).Partner == "Yes"


@pytest.mark.parametrize("field,value", [
    ("Contract", "Three year"),            # unseen category
    ("gender", "female"),                  # wrong case is a different category
    ("MonthlyCharges", -5),                # impossible value
    ("MonthlyCharges", 1e9),               # extreme outlier beyond business range
    ("MonthlyCharges", math.inf),
    ("MonthlyCharges", float("nan")),
    ("tenure", 500),
    ("tenure", "ten"),
    ("SeniorCitizen", 2),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, field: value})


def test_missing_required_field_rejected():
    p = dict(VALID); p.pop("Contract")
    with pytest.raises(ValidationError):
        CustomerFeatures(**p)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "Churn": "Yes"})      # a label sneaking into a scoring request


def test_inconsistent_services_rejected():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID, "InternetService": "No"})   # add-ons say "No", not "No internet service"


def test_batch_limits():
    with pytest.raises(ValidationError):
        BatchRequest(records=[])
    with pytest.raises(ValidationError):
        BatchRequest(records=[VALID] * 1001)
