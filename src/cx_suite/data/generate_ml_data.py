"""Synthetic customer-level dataset for the CX AI/ML use cases.

One row per customer account (lab for Beckman Coulter, ordering practice for
C2N Diagnostics). Two labels are produced:

* ``churn_12m``      - account attrition / disengagement in the next 12 months
                        (supervised early-warning model).
* ``converted_60d``  - account expanded orders within 60 days of a historical,
                        randomized Next-Best-Action outreach (``nba_outreach``),
                        used to train and validate an uplift model.

The generator encodes a known ground truth (``true_uplift``) so model quality
can be checked against reality, something real data never allows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COMPANIES, DATA_RAW, REGIONS, SEED


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 24_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    company = rng.choice(list(COMPANIES), size=n, p=[0.55, 0.45])
    is_c2n = company == "C2N Diagnostics"

    cust_type = np.where(
        is_c2n,
        rng.choice(COMPANIES["C2N Diagnostics"]["customer_types"], size=n, p=[0.38, 0.22, 0.30, 0.10]),
        rng.choice(COMPANIES["Beckman Coulter"]["customer_types"], size=n, p=[0.45, 0.15, 0.20, 0.20]),
    )
    product = np.where(
        is_c2n,
        "PrecivityAD2 Blood Test",
        rng.choice(COMPANIES["Beckman Coulter"]["product_lines"], size=n, p=[0.35, 0.30, 0.20, 0.15]),
    )
    region = rng.choice(REGIONS, size=n, p=[0.24, 0.22, 0.20, 0.14, 0.20])
    tier = rng.choice(["A", "B", "C"], size=n, p=[0.20, 0.35, 0.45])
    tier_num = pd.Series(tier).map({"A": 3, "B": 2, "C": 1}).to_numpy()

    tenure = np.where(is_c2n, rng.gamma(2.0, 9.0, n), rng.gamma(3.0, 22.0, n)).clip(1, 240).round()
    volume = np.where(
        is_c2n,
        rng.lognormal(4.2, 0.8, n) * (0.6 + 0.4 * tier_num),
        rng.lognormal(10.2, 0.9, n) * (0.5 + 0.5 * tier_num),
    ).round()

    downtime = np.where(is_c2n, np.nan, rng.gamma(1.6, 6.0, n))
    tat_days = np.where(is_c2n, rng.normal(8.5, 2.2, n).clip(3, 20), np.nan)
    service_pain = np.where(is_c2n, _z(np.nan_to_num(tat_days, nan=8.5)), _z(np.nan_to_num(downtime, nan=9.6)))

    tickets = rng.poisson(np.exp(0.6 + 0.25 * tier_num + 0.35 * service_pain).clip(0.2, 30))
    resolution_hrs = rng.lognormal(2.8 + 0.15 * service_pain, 0.5, n).round(1)

    email_open = rng.beta(2.2, 5.0, n).round(3)
    portal_logins = rng.poisson(np.exp(1.2 + 0.25 * tier_num + 1.6 * (email_open - 0.3)))
    rep_visits = rng.poisson(0.8 + 1.1 * tier_num)
    webinars = rng.poisson(0.4 + 2.2 * email_open)
    digital_score = (40 * email_open + 3 * np.log1p(portal_logins) + 4 * np.log1p(webinars) + rng.normal(0, 3, n)).clip(0, 60).round(1)

    nps_latent = 7.6 - 0.75 * service_pain - 0.18 * _z(resolution_hrs) + 0.3 * _z(email_open) + rng.normal(0, 1.3, n)
    nps = nps_latent.clip(0, 10).round()
    nps_missing = rng.random(n) < 0.16
    nps_obs = np.where(nps_missing, np.nan, nps)

    competitor = (rng.random(n) < np.where(is_c2n, 0.42, 0.33)).astype(int)
    price_inc = rng.choice([0, 2, 3, 5, 7], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    renewal_days = rng.integers(0, 730, n)
    days_since_order = rng.exponential(np.where(is_c2n, 35, 22) * np.exp(-0.2 * _z(email_open)))
    days_since_order = days_since_order.clip(0, 365).round()
    order_trend = rng.normal(0.02 + 0.03 * _z(nps_latent), 0.12, n).round(3)

    # ---------------- churn ground truth
    churn_logit = (
        -3.20
        + 0.30 * _z(tickets) + 0.30 * _z(resolution_hrs) + 0.45 * service_pain
        - 0.65 * _z(nps_latent) - 0.30 * _z(email_open) - 0.30 * _z(np.log1p(portal_logins))
        - 0.25 * _z(rep_visits) + 0.55 * _z(days_since_order) - 0.45 * _z(order_trend)
        + 0.55 * competitor + 0.10 * price_inc - 0.25 * _z(np.log(tenure))
        + 0.6 * ((renewal_days < 120) & (nps_latent < 6)).astype(float)
        + 0.25 * is_c2n
    )
    churn = (rng.random(n) < _sigmoid(churn_logit)).astype(int)

    # ---------------- randomized historical NBA campaign (uplift ground truth)
    treat = (rng.random(n) < 0.5).astype(int)
    base_logit = -1.9 + 0.35 * _z(digital_score) + 0.25 * (tier_num - 2) + 0.3 * _z(order_trend) - 0.4 * churn
    tau = (
        0.10
        + 0.85 * _sigmoid(-2.0 * _z(email_open)) * (tier_num >= 2)
        + 0.55 * (is_c2n & np.isin(cust_type, ["Primary Care", "Neurology Practice"]) & (tenure < 18))
        - 1.10 * ((nps_latent < 5.5) | ((tickets >= 6) & (service_pain > 1.0)))   # 'sleeping dogs'
    )
    p0 = _sigmoid(base_logit)
    p1 = _sigmoid(base_logit + tau)
    converted = (rng.random(n) < np.where(treat == 1, p1, p0)).astype(int)
    unsub = (rng.random(n) < (0.012 + treat * 0.015 * (email_open < 0.15))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(n)],
        "company": company,
        "customer_type": cust_type,
        "product_line": product,
        "region": region,
        "account_tier": tier,
        "tenure_months": tenure.astype(int),
        "annual_test_volume": volume.astype(int),
        "instrument_downtime_hrs_90d": np.round(downtime, 1),
        "result_turnaround_days": np.round(tat_days, 1),
        "support_tickets_90d": tickets,
        "avg_resolution_hrs": resolution_hrs,
        "nps_score": nps_obs,
        "email_open_rate": email_open,
        "portal_logins_90d": portal_logins,
        "rep_visits_90d": rep_visits,
        "webinars_attended_12m": webinars,
        "digital_engagement_score": digital_score,
        "days_since_last_order": days_since_order.astype(int),
        "order_trend_qoq": order_trend,
        "contract_days_to_renewal": renewal_days,
        "competitor_present": competitor,
        "price_increase_pct": price_inc,
        "nba_outreach": treat,
        "converted_60d": converted,
        "unsubscribed_60d": unsub,
        "churn_12m": churn,
        # ground truth columns (would never exist in real life) for validation only
        "true_uplift": np.round(p1 - p0, 4),
        "true_p_control": np.round(p0, 4),
    })
    return df


def main() -> pd.DataFrame:
    df = generate()
    out = DATA_RAW / "cx_customer_ml_dataset.csv"
    df.to_csv(out, index=False)
    print(f"[ml-data] {len(df):,} rows -> {out.relative_to(out.parents[2])}  "
          f"churn={df.churn_12m.mean():.1%} conv_T={df[df.nba_outreach==1].converted_60d.mean():.1%} "
          f"conv_C={df[df.nba_outreach==0].converted_60d.mean():.1%}")
    return df


if __name__ == "__main__":
    main()
