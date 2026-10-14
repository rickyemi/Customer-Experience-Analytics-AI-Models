"""In-house Marketing Mix Model (MMM) and budget optimizer.

Model per brand (geo-pooled panel: 5 regions x 156 weeks):

    orders_rt = base_rt + sum_c beta_c * w_r * Hill( Adstock(spend_c,rt ; decay_c) ; K_c,r ) + e_rt
    base_rt   = a_r + w_r * [ b1*t + b2*t^2 + Fourier(52w, 2 harmonics) + b_h*holiday_t ]

    w_r = region's share of brand orders; K_c,r = kmult_c x mean adstocked regional spend.

* Adstock captures carry-over (a rep visit or congress keeps paying off for weeks).
* Hill captures diminishing returns (the 10th email in a month adds little).
* Media betas are constrained >= 0 and estimated as a MAP solution under weakly
  informative Gaussian ROI priors (mean 2.0, sd 1.5): bounded least squares with
  prior "pseudo-observation" rows. This stabilizes small-spend channels, the
  classic MMM identifiability problem.
* Carry-over decay is searched within business-plausible ranges per channel
  (field/events carry longer than search), by coordinate descent on a
  time-ordered train split; the last 13 weeks are held out for accuracy.
* Uncertainty: 200-draw moving-block bootstrap of residuals for ROI intervals.

Why in-house instead of a vendor black box: transparent assumptions, weekly
refresh in our own Snowflake/Python stack, governed outputs that feed the same
KPI layer, and direct linkage to brand planning. Bayesian MMM frameworks
(PyMC-Marketing, Google Meridian) are the natural production upgrade path.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from ..config import BRANDS, CHANNELS, DATA_PROCESSED, FIG_ANALYTICS, METRICS, NEUTRAL, PALETTE, SEED, TEXT_2
from ..data.generate_analytics_data import adstock, hill
from ..viz import save

CH = list(CHANNELS)
DECAY_RANGE = {"REP": (0.3, 0.8), "CONF": (0.3, 0.8), "WEB": (0.2, 0.6), "EML": (0.0, 0.4),
               "SRCH": (0.0, 0.3), "DISP": (0.1, 0.5), "SOC": (0.1, 0.5)}
DECAY_STEP = 0.1
PRIOR_ROI_MEAN, PRIOR_ROI_SD = 2.0, 2.5
# Experiment-calibrated priors (ROI mean, sd) from historical geo-holdout lift tests.
# Low-spend channels carry too little week-to-week signal for the MMM alone, so
# results from randomized geo tests are fed in as informative priors (triangulation).
LIFT_TEST_PRIORS = {
    "C2N_AD2": {"EML": (4.9, 0.8), "WEB": (3.3, 0.8)},
    "BEC_HEM": {"EML": (3.5, 0.8)},
    "BEC_IA": {},
}
K_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]          # half-saturation as multiple of mean adstocked spend
S_FIXED = 1.0
HOLDOUT = 13


class BrandMMM:
    """Geo-pooled MMM: region x week panel, shared channel response parameters.

    Pooling five regions multiplies the information available to separate
    channel effects (regional flighting varies independently), while betas,
    decay and saturation stay shared so the model remains parsimonious.
    """

    def __init__(self, brand: str, df: pd.DataFrame):
        self.brand = brand
        self.df = df.sort_values(["region_key", "week_start"]).reset_index(drop=True)
        self.aov = BRANDS[brand][3]
        self.spend = self.df[CH].to_numpy(float)
        self.y = self.df["orders"].to_numpy(float)
        self.regions = self.df.region_key.to_numpy()
        self.weeks = sorted(self.df.week_start.unique())
        self.T = len(self.weeks)
        self.widx = self.df.groupby("region_key").cumcount().to_numpy()
        share = self.df.groupby("region_key").orders.sum() / self.df.orders.sum()
        self.w = self.df.region_key.map(share).to_numpy()
        self.blocks = [np.where(self.regions == r)[0] for r in sorted(np.unique(self.regions))]
        self.C = self._controls()
        self.decay = {c: float(np.mean(DECAY_RANGE[c])) for c in CH}
        self.kmult = {c: 1.0 for c in CH}

    def _controls(self) -> np.ndarray:
        t = self.widx / 52
        hol = self.df.is_holiday_week.to_numpy(float)
        shape = [t, t ** 2, hol]
        for h in (1, 2):
            shape += [np.sin(2 * np.pi * h * self.widx / 52), np.cos(2 * np.pi * h * self.widx / 52)]
        dummies = [(self.regions == r).astype(float) for r in sorted(np.unique(self.regions))]
        return np.column_stack(dummies + [self.w * c for c in shape])

    # ------------------------------------------------------------ transforms
    def media_matrix(self) -> np.ndarray:
        M = np.zeros((len(self.y), len(CH)))
        for b in self.blocks:
            for j, c in enumerate(CH):
                ad = adstock(self.spend[b, j], self.decay[c])
                M[b, j] = self.w[b] * hill(ad, self.kmult[c] * ad.mean(), S_FIXED)
        return M

    def _design(self, M):
        return np.column_stack([M, self.C])

    def _fit_coef(self, X, y, idx=None):
        """MAP estimate: bounded least squares + Gaussian prior rows on media betas (ROI scale)."""
        nm = len(CH)
        lb = np.r_[np.zeros(nm), -np.inf * np.ones(X.shape[1] - nm)]
        ub = np.inf * np.ones(X.shape[1])
        if getattr(self, "sigma_e", None) is None:
            b0 = lsq_linear(X, y, bounds=(lb, ub), lsmr_tol="auto").x
            self.sigma_e = float(np.std(y - X @ b0))
        sp = self.spend if idx is None else self.spend[idx]
        # beta_c (orders at saturation) implied by ROI r: beta = r * sum(spend) / (aov * sum(h_c))
        scale = sp.sum(axis=0) / (self.aov * np.maximum(X[:, :nm].sum(axis=0), 1e-9))
        pri = LIFT_TEST_PRIORS.get(self.brand, {})
        mu = np.array([pri.get(c, (PRIOR_ROI_MEAN, PRIOR_ROI_SD))[0] for c in CH])
        sd = np.array([pri.get(c, (PRIOR_ROI_MEAN, PRIOR_ROI_SD))[1] for c in CH])
        beta_prior, beta_sd = mu * scale, sd * scale
        P = np.zeros((nm, X.shape[1]))
        P[np.arange(nm), np.arange(nm)] = self.sigma_e / beta_sd
        Xa = np.vstack([X, P])
        ya = np.r_[y, self.sigma_e * beta_prior / beta_sd]
        return lsq_linear(Xa, ya, bounds=(lb, ub), lsmr_tol="auto").x

    def _sse(self, idx):
        X = self._design(self.media_matrix())
        coef = self._fit_coef(X[idx], self.y[idx], idx)
        return np.sum((self.y[idx] - X[idx] @ coef) ** 2)

    # ------------------------------------------------------------ fit
    def tune(self, idx, passes: int = 2):
        for _ in range(passes):
            for c in CH:
                best = (np.inf, self.decay[c], self.kmult[c])
                lo, hi = DECAY_RANGE[c]
                for d in np.round(np.arange(lo, hi + 1e-9, DECAY_STEP), 2):
                    for k in K_GRID:
                        self.decay[c], self.kmult[c] = d, k
                        e = self._sse(idx)
                        if e < best[0]:
                            best = (e, d, k)
                self.decay[c], self.kmult[c] = best[1], best[2]
        return self

    def _national(self, v):
        return pd.Series(v).groupby(self.widx).sum().to_numpy()

    def fit(self):
        tr = np.where(self.widx < self.T - HOLDOUT)[0]
        np.where(self.widx >= self.T - HOLDOUT)[0]
        self.tune(tr)
        X = self._design(self.media_matrix())
        coef_tr = self._fit_coef(X[tr], self.y[tr], tr)
        pred = X @ coef_tr
        y_nat, p_nat = self._national(self.y), self._national(pred)
        ntr = self.T - HOLDOUT
        self.holdout = {
            "mape": float(np.mean(np.abs(y_nat[ntr:] - p_nat[ntr:]) / y_nat[ntr:])),
            "r2_train": float(1 - np.sum((y_nat[:ntr] - p_nat[:ntr]) ** 2) / np.sum((y_nat[:ntr] - y_nat[:ntr].mean()) ** 2)),
            "geo_r2_train": float(1 - np.sum((self.y[tr] - pred[tr]) ** 2) / np.sum((self.y[tr] - self.y[tr].mean()) ** 2)),
        }
        self.coef = self._fit_coef(X, self.y)             # refit on full history for decomposition
        self.X = X
        self.fitted = X @ self.coef
        self.y_nat, self.pred_holdout_trainfit = y_nat, p_nat
        self.week_start = pd.Series(self.weeks)
        self.beta = dict(zip(CH, self.coef[: len(CH)]))
        f_nat = self._national(self.fitted)
        self.r2 = float(1 - np.sum((y_nat - f_nat) ** 2) / np.sum((y_nat - y_nat.mean()) ** 2))
        self.nat_spend = np.column_stack([self._national(self.spend[:, j]) for j in range(len(CH))])
        return self

    def bootstrap_roi(self, n_boot: int = 200, block: int = 8):
        rng = np.random.default_rng(SEED)
        resid = self.y - self.fitted
        rois = []
        for _ in range(n_boot):
            r = np.empty_like(resid)
            for b in self.blocks:
                rb = resid[b]
                starts = rng.integers(0, len(rb) - block, len(rb) // block + 1)
                r[b] = np.concatenate([rb[s:s + block] for s in starts])[: len(rb)]
            rois.append(self._roi(self._fit_coef(self.X, self.fitted + r)))
        rois = np.array(rois)
        return {c: (float(np.percentile(rois[:, j], 5)), float(np.percentile(rois[:, j], 95))) for j, c in enumerate(CH)}

    # ------------------------------------------------------------ outputs
    def contributions(self, coef=None):
        """National weekly contributions by driver."""
        coef = self.coef if coef is None else coef
        M = self.X[:, : len(CH)] * coef[: len(CH)]
        out = pd.DataFrame({c: self._national(M[:, j]) for j, c in enumerate(CH)})
        out["Base"] = self._national(self.X[:, len(CH):] @ coef[len(CH):])
        return out

    def _roi(self, coef):
        M = self.X[:, : len(CH)] * coef[: len(CH)]
        return M.sum(axis=0) * self.aov / self.spend.sum(axis=0)

    def roi(self):
        return dict(zip(CH, self._roi(self.coef).tolist()))

    def steady_state_response(self, c: str, weekly_spend):
        """National incremental orders/week at a sustained weekly spend level
        (allocated across regions in today's proportions)."""
        j = CH.index(c)
        ref = adstock(self.nat_spend[:, j], self.decay[c]).mean()
        ad = np.asarray(weekly_spend) / (1 - self.decay[c])
        return self.beta[c] * hill(ad, self.kmult[c] * ref, S_FIXED)

    def marginal_roi(self):
        out = {}
        for j, c in enumerate(CH):
            s_ = self.nat_spend[-52:, j].mean()
            d = 0.01 * s_
            out[c] = float((self.steady_state_response(c, s_ + d) - self.steady_state_response(c, s_)) * self.aov / d)
        return out

    def optimize(self, lo: float = 0.6, hi: float = 1.5, rep_lo: float = 0.85):
        """Reallocate the last-52-week average budget at constant total spend.
        Guardrails: each channel within [lo, hi] x current; field force cannot drop
        more than 15% in one cycle (headcount constraint)."""
        cur = self.nat_spend[-52:].mean(axis=0)
        total = cur.sum()

        def neg_rev(x):
            return -sum(self.steady_state_response(c, x[j]) for j, c in enumerate(CH)) * self.aov

        bounds = [((rep_lo if c == "REP" else lo) * cur[j], hi * cur[j]) for j, c in enumerate(CH)]
        res = minimize(neg_rev, cur, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - total}])
        cur_rev, opt_rev = -neg_rev(cur), -neg_rev(res.x)
        return {"current": dict(zip(CH, cur.tolist())), "optimal": dict(zip(CH, res.x.tolist())),
                "incremental_revenue_current_wk": float(cur_rev), "incremental_revenue_optimal_wk": float(opt_rev),
                "uplift_pct": float(opt_rev / cur_rev - 1), "annualized_gain_usd": float((opt_rev - cur_rev) * 52)}


def run(mf: pd.DataFrame) -> dict:
    truth = json.loads((DATA_PROCESSED / "mmm_ground_truth.json").read_text())
    models, summary = {}, {}
    for b in BRANDS:
        m = BrandMMM(b, mf[mf.brand_code == b]).fit()
        ci = m.bootstrap_roi()
        opt = m.optimize()
        models[b] = m
        summary[b] = {
            "r2_full": m.r2, **m.holdout, "roi": m.roi(), "roi_90ci": ci, "marginal_roi": m.marginal_roi(),
            "true_roi": {c: truth[b]["channels"][c]["true_roi"] for c in CH},
            "lift_test_priors": LIFT_TEST_PRIORS.get(b, {}),
            "decay_est": m.decay, "decay_true": {c: truth[b]["channels"][c]["decay"] for c in CH},
            "optimizer": opt,
            "media_share_of_orders": float(m.contributions()[CH].sum().sum() / m.fitted.sum()),
            "n_obs_geo_weeks": int(len(m.y)),
            "total_spend": float(m.spend.sum()),
            "incremental_revenue": float(m.contributions()[CH].sum().sum() * m.aov),
        }
        print(f"[mmm] {b}: R2={m.r2:.3f} holdout MAPE={m.holdout['mape']:.1%} "
              f"optimizer +{opt['uplift_pct']:.1%} incremental revenue")
    tot_inc = sum(s["incremental_revenue"] for s in summary.values())
    tot_sp = sum(s["total_spend"] for s in summary.values())
    summary["portfolio"] = {"incremental_roi": tot_inc / tot_sp,
                            "annualized_reallocation_gain_usd": sum(summary[b]["optimizer"]["annualized_gain_usd"] for b in BRANDS)}
    (METRICS / "mmm_results.json").write_text(json.dumps(summary, indent=2))
    figures(models, summary, mf)
    return {"models": models, "summary": summary}


# ---------------------------------------------------------------- figures
def figures(models: dict, summary: dict, mf: pd.DataFrame) -> None:
    names = {c: CHANNELS[c][0] for c in CH}
    brands = list(BRANDS)

    # 05 fit + holdout
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, b in zip(axes, brands):
        m = models[b]
        wk = m.week_start
        ax.plot(wk, m.y_nat, color=NEUTRAL, lw=1.4, label="Actual orders")
        ax.plot(wk, m.pred_holdout_trainfit, color=PALETTE[0], lw=1.8, label="Model (fit on train only)")
        ax.axvspan(wk.iloc[-HOLDOUT], wk.iloc[-1], color=PALETTE[3], alpha=0.12, lw=0)
        ax.text(wk.iloc[-HOLDOUT], ax.get_ylim()[1] * 0.97, " hold-out", va="top", color=TEXT_2, fontsize=8.5)
        ax.set_title(f"{BRANDS[b][0]}  |  R² {summary[b]['r2_train']:.2f} (train), hold-out MAPE "
                     f"{summary[b]['mape']:.1%}", fontsize=11)
        ax.set_ylabel("Orders / week")
    axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle("MMM fit: actual vs predicted weekly orders", x=0.01, ha="left", fontsize=13, weight="bold")
    save(fig, FIG_ANALYTICS / "05_mmm_fit_holdout.png")

    # 06 decomposition for C2N (the growth brand)
    b = "C2N_AD2"
    m = models[b]
    contrib = m.contributions()
    contrib.index = m.week_start
    order = ["Base"] + sorted(CH, key=lambda c: -contrib[c].sum())
    cols = [NEUTRAL] + PALETTE[: len(CH)]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.stackplot(contrib.index, *[contrib[c].clip(lower=0) for c in order], colors=cols,
                 labels=["Base (trend, seasonality)"] + [names[c] for c in order[1:]], edgecolor="white", linewidth=0.3)
    ax.plot(m.week_start, m.y_nat, color="black", lw=0.9, label="Actual orders")
    ax.set(title=f"{BRANDS[b][0]}: weekly orders decomposed by driver", ylabel="Orders / week",
           ylim=(0, m.y_nat.max() * 1.28))
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    save(fig, FIG_ANALYTICS / "06_mmm_contribution_decomposition.png",
         f"Media drives {summary[b]['media_share_of_orders']:.0%} of fitted orders over FY23-FY25; the remainder is organic base and growth trend.")

    # 07 ROI estimated vs true (validation) per brand
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, b in zip(axes, brands):
        s = summary[b]
        est = np.array([s["roi"][c] for c in CH])
        lo = np.array([s["roi_90ci"][c][0] for c in CH])
        hi = np.array([s["roi_90ci"][c][1] for c in CH])
        tru = np.array([s["true_roi"][c] for c in CH])
        y = np.arange(len(CH))
        ax.barh(y, est, color=PALETTE[0], height=0.55, label="MMM estimate (90% CI)")
        ax.errorbar(est, y, xerr=[np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)], fmt="none", ecolor=TEXT_2, lw=1, capsize=3)
        ax.scatter(tru, y, marker="D", s=36, color=PALETTE[1], zorder=3, label="True ROI (synthetic)")
        for yi, c in zip(y, CH):
            if c in LIFT_TEST_PRIORS.get(b, {}):
                ax.text(max(hi[yi], tru[yi]) + 0.15, yi, "lift-test prior", va="center", fontsize=7.5, color=TEXT_2)
        ax.axvline(1, color=NEUTRAL, ls="--", lw=1)
        ax.set_yticks(y, [names[c] for c in CH])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set(title=BRANDS[b][0], xlabel="Incremental revenue per $1")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Channel ROI: in-house MMM estimate vs known truth", x=0.01, ha="left", fontsize=13, weight="bold")
    save(fig, FIG_ANALYTICS / "07_mmm_roi_vs_truth.png", "Dashed line = break-even (ROI 1.0). Truth is knowable only because data are simulated. Low-spend channels use geo lift-test priors.")

    # 08 response curves (C2N)
    m = models["C2N_AD2"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for j, c in enumerate(CH):
        cur = m.nat_spend[-52:, j].mean()
        grid = np.linspace(0, cur * 2.5, 80)
        rev = m.steady_state_response(c, grid) * m.aov
        ax.plot(grid / 1e3, rev / 1e3, color=PALETTE[j], label=names[c])
        ax.scatter([cur / 1e3], [m.steady_state_response(c, cur) * m.aov / 1e3], color=PALETTE[j], s=40,
                   edgecolor="white", linewidth=1.5, zorder=3)
    ax.set(title="PrecivityAD2: response curves (dots = current weekly spend)",
           xlabel="Sustained weekly spend ($K)", ylabel="Incremental revenue per week ($K)")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    save(fig, FIG_ANALYTICS / "08_mmm_response_curves_c2n.png", "Curves flatten where extra spend buys little: the signal the optimizer acts on.")

    # 09 budget reallocation per brand
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, b in zip(axes, brands):
        o = summary[b]["optimizer"]
        chg = np.array([o["optimal"][c] / o["current"][c] - 1 for c in CH]) * 100
        y = np.arange(len(CH))
        ax.barh(y, chg, color=[PALETTE[0] if v >= 0 else PALETTE[1] for v in chg], height=0.55)
        for yi, v in zip(y, chg):
            ax.text(v + (2 if v >= 0 else -2), yi, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                    color=TEXT_2, fontsize=8.5)
        ax.axvline(0, color=TEXT_2, lw=0.8)
        ax.set_yticks(y, [names[c] for c in CH])
        ax.set_xlim(-60, 75)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set(title=f"{BRANDS[b][0]}\n+{o['uplift_pct']:.1%} incremental revenue, same budget", xlabel="Change vs current spend")
    fig.suptitle("Budget reallocation scenario (constant total spend)", x=0.01, ha="left", fontsize=13, weight="bold")
    save(fig, FIG_ANALYTICS / "09_budget_reallocation.png",
         "Constraints: each channel 0.6x to 1.5x current; field force at most -15% per cycle. Directional, to be validated with geo tests.")
