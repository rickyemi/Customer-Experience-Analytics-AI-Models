"""Feature engineering and a reusable sklearn preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["company", "customer_type", "region", "account_tier"]
NUMERIC = [
    "tenure_months", "log_test_volume", "service_pain_index", "support_tickets_90d",
    "avg_resolution_hrs", "nps_score", "nps_missing", "email_open_rate", "portal_logins_90d",
    "rep_visits_90d", "webinars_attended_12m", "digital_engagement_score", "days_since_last_order",
    "order_trend_qoq", "contract_days_to_renewal", "renewal_within_120d", "competitor_present",
    "price_increase_pct",
]
# Columns that must never enter a model (labels, treatment, leakage, ground truth)
EXCLUDE = ["customer_id", "churn_12m", "converted_60d", "unsubscribed_60d", "nba_outreach", "true_uplift", "true_p_control"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features. Pure function: same logic in training and scoring."""
    out = df.copy()
    out["log_test_volume"] = np.log1p(out["annual_test_volume"])
    # One service-quality signal across both businesses: instrument downtime (BEC)
    # and result turnaround (C2N), each standardized within its own company.
    dt = out["instrument_downtime_hrs_90d"]
    tat = out["result_turnaround_days"]
    z_dt = (dt - dt.mean()) / dt.std()
    z_tat = (tat - tat.mean()) / tat.std()
    out["service_pain_index"] = z_dt.fillna(z_tat).fillna(0.0)
    out["nps_missing"] = out["nps_score"].isna().astype(int)
    out["renewal_within_120d"] = (out["contract_days_to_renewal"] < 120).astype(int)
    return out


def preprocessor(scale: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return engineer(df)[NUMERIC + CATEGORICAL]
