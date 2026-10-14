"""Use case 1: Account Attrition Early-Warning (churn propensity).

Business question: which lab / practice accounts are likely to disengage in
the next 12 months, why, and which ones should Customer Success contact first?

Design: interpretable baseline (logistic regression) vs. gradient boosting,
Platt (sigmoid) calibration so scores read as probabilities, capacity-based
threshold (top decile = CSM capacity), subgroup performance checks and
per-account reason codes for field consumption.
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ..config import DATA_PROCESSED, DATA_RAW, FIG_ML, METRICS, MODELS, NEUTRAL, PALETTE, SEED, TEXT_2
from ..viz import save
from .features import NUMERIC, feature_frame, preprocessor

TARGET = "churn_12m"
CAPACITY = 0.10   # Customer Success can proactively work 10% of accounts per cycle


def lift_at(y, p, frac):
    k = int(len(p) * frac)
    idx = np.argsort(-p)[:k]
    return y[idx].mean() / y.mean()


def reason_codes(model, X: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Occlusion-style local attribution in log-odds space: reset each numeric feature
    to its population median and record how much the account's risk logit falls.
    Reported as each driver's share of the total upward push, for field-friendly text."""
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    base = logit(model.predict_proba(X)[:, 1])
    med = X[NUMERIC].median()
    deltas = {}
    for f in NUMERIC:
        Xm = X.copy()
        Xm[f] = med[f]
        deltas[f] = base - logit(model.predict_proba(Xm)[:, 1])
    D = pd.DataFrame(deltas, index=X.index).clip(lower=0)
    share = D.div(D.sum(axis=1) + 1e-9, axis=0)
    return ["; ".join(f"{c} ({v:.0%})" for c, v in row.nlargest(top_n).items() if v > 0.02)
            for _, row in share.iterrows()]


def run(df: pd.DataFrame | None = None) -> dict:
    df = df if df is not None else pd.read_csv(DATA_RAW / "cx_customer_ml_dataset.csv")
    X = feature_frame(df)
    y = df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=0.3, stratify=y, random_state=SEED)

    models = {
        "Logistic regression": Pipeline([("prep", preprocessor()),
                                         ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "Gradient boosting": Pipeline([("prep", preprocessor(scale=False)),
                                       ("clf", HistGradientBoostingClassifier(
                                           learning_rate=0.05, max_iter=350, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=SEED))]),
    }
    from sklearn.base import clone
    models_raw = {k: clone(v) for k, v in models.items()}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, probs = {}, {}
    for name, m in models.items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
        cal = CalibratedClassifierCV(m, method="sigmoid", cv=3).fit(X_tr, y_tr)
        p = cal.predict_proba(X_te)[:, 1]
        probs[name] = p
        models[name] = cal
        results[name] = {
            "cv_auc_mean": float(cv_auc.mean()), "cv_auc_sd": float(cv_auc.std()),
            "test_auc": float(roc_auc_score(y_te, p)), "test_pr_auc": float(average_precision_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)), "lift_top10": float(lift_at(y_te, p, 0.10)),
            "recall_top10": float(y_te[np.argsort(-p)[: int(len(p) * 0.1)]].sum() / y_te.sum()),
        }
    champion = max(results, key=lambda k: results[k]["test_auc"])
    p = probs[champion]
    model = models[champion]
    joblib.dump(model, MODELS / "churn_model.joblib")

    # ----- subgroup performance (responsible AI check)
    sub = []
    for col in ("company", "region", "account_tier"):
        for g, idx in df_te.groupby(col).groups.items():
            mask = df_te.index.isin(idx)
            sub.append({"group": f"{col}={g}", "auc": roc_auc_score(y_te[mask], p[mask]), "n": int(mask.sum())})
    sub = pd.DataFrame(sub)

    # ----- figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (name, pp) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y_te, pp)
        axes[0].plot(fpr, tpr, color=PALETTE[i], label=f"{name}  AUC {results[name]['test_auc']:.3f}")
        pr, rc, _ = precision_recall_curve(y_te, pp)
        axes[1].plot(rc, pr, color=PALETTE[i], label=f"{name}  AP {results[name]['test_pr_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1)
    axes[1].axhline(y_te.mean(), ls="--", color=NEUTRAL, lw=1, label=f"Base rate {y_te.mean():.1%}")
    axes[0].set(title="ROC curve (hold-out)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall (hold-out)", xlabel="Recall", ylabel="Precision")
    for a in axes:
        a.legend(loc="lower right" if a is axes[0] else "upper right")
    save(fig, FIG_ML / "01_churn_roc_pr.png", "Synthetic data. 30% stratified hold-out, Platt-calibrated models.")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for i, (name, pp) in enumerate(probs.items()):
        fy, fx = calibration_curve(y_te, pp, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", ms=5, color=PALETTE[i], label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1, label="Perfect calibration")
    ax.set(title="Calibration: predicted vs observed churn", xlabel="Predicted probability",
           ylabel="Observed churn rate", xlim=(0, 0.8), ylim=(0, 0.8))
    ax.legend()
    save(fig, FIG_ML / "02_churn_calibration.png")

    order = np.argsort(-p)
    cum = np.cumsum(y_te[order]) / y_te.sum()
    frac = np.arange(1, len(p) + 1) / len(p)
    dec = pd.qcut(pd.Series(p).rank(method="first", ascending=False), 10, labels=range(1, 11))
    dec_rate = pd.Series(y_te).groupby(dec.values, observed=True).mean() / y_te.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(frac, cum, color=PALETTE[0], label=champion)
    axes[0].plot([0, 1], [0, 1], ls="--", color=NEUTRAL, lw=1, label="Random targeting")
    axes[0].axvline(CAPACITY, color=PALETTE[1], lw=1.2, ls=":")
    axes[0].text(CAPACITY + 0.02, 0.08, f"CSM capacity (top {CAPACITY:.0%})\ncaptures {cum[int(len(p)*CAPACITY)]:.0%} of churners",
                 color=TEXT_2, fontsize=9)
    axes[0].set(title="Cumulative gains", xlabel="Share of accounts contacted (ranked by risk)",
                ylabel="Share of churners captured")
    axes[0].legend(loc="lower right")
    axes[1].bar(dec_rate.index.astype(int), dec_rate.values, color=PALETTE[0], width=0.7)
    axes[1].axhline(1, color=NEUTRAL, ls="--", lw=1)
    for x_, v in zip(dec_rate.index.astype(int), dec_rate.values):
        if x_ in (1, 10):
            axes[1].text(x_, v + 0.05, f"{v:.1f}x", ha="center", color=TEXT_2, fontsize=9)
    axes[1].set(title="Lift by risk decile", xlabel="Risk decile (1 = highest)", ylabel="Churn rate vs average")
    axes[1].set_xticks(range(1, 11))
    save(fig, FIG_ML / "03_churn_gains_lift.png")

    pi = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp.index, imp.values, color=PALETTE[0], height=0.65)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set(title="Top churn drivers (permutation importance)", xlabel="Drop in AUC when feature is shuffled")
    save(fig, FIG_ML / "04_churn_feature_importance.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sub_s = sub.sort_values("auc")
    ax.barh(sub_s.group, sub_s.auc, color=PALETTE[0], height=0.6)
    ax.axvline(results[champion]["test_auc"], color=PALETTE[1], ls="--", lw=1.2, label="Overall AUC")
    ax.axvline(results[champion]["test_auc"] - 0.05, color=NEUTRAL, ls=":", lw=1.2, label="Tolerance (-0.05)")
    ax.set(xlim=(0.5, 1.0), title="Subgroup performance check", xlabel="Hold-out AUC")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, FIG_ML / "05_churn_subgroup_auc.png", "Responsible-AI gate: no subgroup may fall more than 0.05 AUC below overall.")

    # ----- operational output: ranked worklist with reason codes
    scored = df_te[["customer_id", "company", "customer_type", "region", "account_tier"]].copy()
    scored["churn_risk"] = p.round(4)
    top = scored.sort_values("churn_risk", ascending=False).head(250).copy()
    raw = models_raw[champion].fit(X_tr, y_tr)   # uncalibrated twin for smooth local attributions
    top["top_reasons"] = reason_codes(raw, X_te.loc[top.index])
    top.to_csv(DATA_PROCESSED / "churn_worklist_top250.csv", index=False)

    out = {"champion": champion, "models": results, "capacity": CAPACITY,
           "subgroup_auc_min": float(sub.auc.min()), "subgroup_auc_max": float(sub.auc.max()),
           "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "base_rate": float(y.mean())}
    (METRICS / "churn_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"[churn] champion={champion} AUC={results[champion]['test_auc']:.3f} "
          f"lift@10={results[champion]['lift_top10']:.2f}")
    return out


if __name__ == "__main__":
    run()
