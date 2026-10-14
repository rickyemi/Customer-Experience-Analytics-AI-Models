"""Use case 2: Next-Best-Action (NBA) outreach targeting with uplift modeling.

Business question: *who should receive proactive outreach* (clinical-education
email + rep follow-up) so that it changes behavior, rather than who will
order anyway? A response model targets "sure things"; an uplift model targets
"persuadables" and avoids "sleeping dogs" (accounts annoyed by outreach).

Training data comes from a historical campaign that was randomized 50/50,
which makes the treatment effect identifiable. Method: T-learner with
gradient boosting, evaluated by Qini / uplift-by-decile on a hold-out,
and (synthetic-only) against the true individual uplift.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DATA_RAW, FIG_ML, METRICS, MODELS, NEUTRAL, PALETTE, SEED, TEXT_2
from ..viz import save
from .estimators import TLearner
from .features import feature_frame

TARGET_SHARE = 0.30   # campaign capacity: 30% of accounts


def qini_curve(y, t, u, bins: int = 50):
    order = np.argsort(-u)
    y, t = y[order], t[order]
    n = len(y)
    xs, qs = [0.0], [0.0]
    for k in np.linspace(n / bins, n, bins).astype(int):
        yt, yc = y[:k][t[:k] == 1].sum(), y[:k][t[:k] == 0].sum()
        nt, nc = (t[:k] == 1).sum(), (t[:k] == 0).sum()
        qs.append(yt - yc * nt / max(nc, 1))
        xs.append(k / n)
    return np.array(xs), np.array(qs)


def run(df: pd.DataFrame | None = None) -> dict:
    df = df if df is not None else pd.read_csv(DATA_RAW / "cx_customer_ml_dataset.csv")
    X = feature_frame(df)
    t = df["nba_outreach"].to_numpy()
    y = df["converted_60d"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.35, stratify=t * 2 + y, random_state=SEED)

    model = TLearner().fit(X.iloc[idx_tr], t[idx_tr], y[idx_tr])
    joblib.dump(model, MODELS / "nba_uplift_tlearner.joblib")
    u = model.predict_uplift(X.iloc[idx_te])
    y_te, t_te = y[idx_te], t[idx_te]
    true_u = df["true_uplift"].to_numpy()[idx_te]

    # Qini and AUUC-style coefficient
    xs, qs = qini_curve(y_te, t_te, u)
    rand = xs * qs[-1]
    qini_coef = float(np.trapezoid(qs - rand, xs) / max(np.trapezoid(np.abs(rand), xs), 1e-9))
    xs_o, qs_o = qini_curve(y_te, t_te, true_u)   # oracle (synthetic only)

    # Uplift by predicted decile: observed (T - C) vs predicted
    dec = pd.qcut(pd.Series(u).rank(method="first", ascending=False), 10, labels=range(1, 11)).to_numpy()
    rows = []
    for d in range(1, 11):
        m = dec == d
        obs = y_te[m & (t_te == 1)].mean() - y_te[m & (t_te == 0)].mean()
        rows.append({"decile": d, "observed_uplift": obs, "predicted_uplift": u[m].mean(), "true_uplift": true_u[m].mean()})
    dec_df = pd.DataFrame(rows)

    # Policy value: expected incremental conversions per 1,000 targeted accounts
    k = int(len(u) * TARGET_SHARE)
    top = np.argsort(-u)[:k]
    rnd = np.random.default_rng(SEED).choice(len(u), k, replace=False)
    policy = {
        "uplift_model_top30_true_uplift": float(true_u[top].mean()),
        "random_30_true_uplift": float(true_u[rnd].mean()),
        "response_model_note": "see pilot_design for head-to-head simulation",
    }
    corr = float(np.corrcoef(u, true_u)[0, 1])

    # ----- figures
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(xs * 100, qs, color=PALETTE[0], label=f"T-learner (Qini coef {qini_coef:.2f})")
    axes[0].plot(xs_o * 100, qs_o, color=PALETTE[2], ls="-.", lw=1.5, label="Oracle (true uplift)")
    axes[0].plot(xs * 100, rand, color=NEUTRAL, ls="--", lw=1, label="Random targeting")
    axes[0].axvline(TARGET_SHARE * 100, color=PALETTE[1], ls=":", lw=1.2)
    axes[0].text(TARGET_SHARE * 100 + 1.5, qs.max() * 1.02, "Campaign capacity 30%", color=TEXT_2, fontsize=9)
    axes[0].set(title="Qini curve (hold-out)", xlabel="% of accounts targeted (ranked by predicted uplift)",
                ylabel="Incremental conversions")
    axes[0].legend(loc="lower right")
    w = 0.38
    axes[1].bar(dec_df.decile - w / 2, dec_df.observed_uplift * 100, width=w, color=PALETTE[0], label="Observed (T minus C)")
    axes[1].bar(dec_df.decile + w / 2, dec_df.predicted_uplift * 100, width=w, color=PALETTE[1], label="Predicted")
    axes[1].axhline(0, color=TEXT_2, lw=0.8)
    axes[1].set(title="Uplift by predicted decile", xlabel="Predicted-uplift decile (1 = most persuadable)",
                ylabel="Conversion uplift (pts)")
    axes[1].set_xticks(range(1, 11))
    axes[1].legend()
    save(fig, FIG_ML / "06_nba_uplift_qini_deciles.png",
         "Training data: historical 50/50 randomized outreach. Bottom deciles flag 'sleeping dogs' to suppress.")

    out = {"qini_coefficient": qini_coef, "corr_pred_vs_true_uplift": corr, "deciles": rows,
           "policy": policy, "target_share": TARGET_SHARE,
           "ate_observed": float(y_te[t_te == 1].mean() - y_te[t_te == 0].mean())}
    (METRICS / "uplift_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"[uplift] qini={qini_coef:.2f} corr(true)={corr:.2f} top-decile obs uplift="
          f"{rows[0]['observed_uplift']:.3f} bottom={rows[-1]['observed_uplift']:.3f}")
    return out


if __name__ == "__main__":
    run()
