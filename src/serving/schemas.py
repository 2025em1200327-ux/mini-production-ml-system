"""Pydantic contracts for the API and the batch CLI. Invalid payloads are rejected at the boundary
with a 422 and a field-level reason, before they can reach the model."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
MAX_BATCH = 1000


class CustomerFeatures(BaseModel):
    """One customer at scoring time. Field names match the training columns exactly."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_id: Optional[str] = Field(None, max_length=64, description="Echoed back; not used by the model")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0, le=1000, allow_inf_nan=False)
    TotalCharges: Optional[float] = Field(None, ge=0, le=200_000, allow_inf_nan=False,
                                          description="May be missing for new customers; imputed by the pipeline")

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, v):
        """Upstream systems often pad values (' Yes '); trim before matching the allowed categories."""
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def services_consistent(self):
        """Cross-field rules that always hold in the source system; a violation means a broken upstream feed."""
        addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        no_net = self.InternetService == "No"
        for a in addons:
            if no_net != (getattr(self, a) == "No internet service"):
                raise ValueError(f"{a}='{getattr(self, a)}' is inconsistent with InternetService='{self.InternetService}'")
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError(f"MultipleLines='{self.MultipleLines}' is inconsistent with PhoneService='{self.PhoneService}'")
        return self


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    churn_flag: bool
    decision: Literal["offer_retention", "no_action"]
    threshold: float
    model_version: str
    degraded: bool = Field(description="True when the heuristic fallback produced this score")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    n: int
    model_version: str
    degraded: bool
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    fallback_available: bool


class ErrorResponse(BaseModel):
    error: str
    detail: list | str
    request_id: Optional[str] = None
