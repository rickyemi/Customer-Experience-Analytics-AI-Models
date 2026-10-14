"""Post-deployment monitoring: Population Stability Index (PSI) for feature and
score drift, with retraining triggers.

Thresholds (industry convention): PSI < 0.10 stable, 0.10-0.25 investigate,
> 0.25 significant drift -> trigger retraining review with model governance.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import DATA_RAW, FIG_ML, METRICS, MODELS, PALETTE, SEED, TEXT_2
from ..data.generate_ml_data import generate
from ..viz import save
from .features import NUMERIC, engineer, feature_frame


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def simulate_next_quarter(n: int = 8000) -> pd.DataFrame:
    """Scoring batch with realistic drift: email fatigue, a service backlog after an
    instrument software update, and a growing C2N mix."""
    df = generate(n=n, seed=SEED + 99)
    rng = np.random.default_rng(SEED + 99)
    df["email_open_rate"] = (df.email_open_rate * 0.72).round(3)
    df["support_tickets_90d"] = rng.poisson(df.support_tickets_90d * 1.35 + 0.3)
    df["avg_resolution_hrs"] = (df.avg_resolution_hrs * 1.25).round(1)
    return df


def run() -> dict:
    train = pd.read_csv(DATA_RAW / "cx_customer_ml_dataset.csv")
    cur = simulate_next_quarter()
    et, ec = engineer(train), engineer(cur)
    vals = {f: psi(et[f], ec[f]) for f in NUMERIC if et[f].nunique() > 2}
    model = joblib.load(MODELS / "churn_model.joblib")
    vals["MODEL SCORE (churn risk)"] = psi(model.predict_proba(feature_frame(train))[:, 1],
                                          model.predict_proba(feature_frame(cur))[:, 1])
    s = pd.Series(vals).sort_values()
    colors = [PALETTE[7] if v > 0.25 else PALETTE[3] if v > 0.10 else PALETTE[0] for v in s.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(s.index, s.values, color=colors, height=0.65)
    ax.axvline(0.10, color=TEXT_2, ls=":", lw=1)
    ax.axvline(0.25, color=TEXT_2, ls="--", lw=1)
    ax.text(0.105, -0.9, "investigate", color=TEXT_2, fontsize=8)
    ax.text(0.255, -0.9, "retrain review", color=TEXT_2, fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set(title="Drift monitor: training vs next-quarter scoring batch", xlabel="Population Stability Index (PSI)")
    for y_, v in enumerate(s.values):
        if v > 0.10:
            ax.text(v + 0.005, y_, f"{v:.2f}", va="center", color=TEXT_2, fontsize=8)
    save(fig, FIG_ML / "08_drift_monitor_psi.png",
         "Bars: blue stable (<0.10), yellow investigate (0.10-0.25), red significant (>0.25). Simulated drift scenario.")
    out = {"psi": {k: round(v, 4) for k, v in vals.items()},
           "features_over_0_25": [k for k, v in vals.items() if v > 0.25],
           "action": "Open retraining ticket; validate on fresh labels; champion/challenger before swap"}
    (METRICS / "drift_psi.json").write_text(json.dumps(out, indent=2))
    print(f"[drift] significant: {out['features_over_0_25']}")
    return out


if __name__ == "__main__":
    run()
