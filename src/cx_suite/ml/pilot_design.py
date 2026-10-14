"""MVP pilot: A/B test design, power analysis and simulated readout for the
uplift-guided Next-Best-Action program.

Pre-registered design
---------------------
* Unit           : account (randomized; stratified by company x tier)
* Arm A (control): business-as-usual rule, i.e. contact the 30% most digitally
                   engaged accounts (today's segmentation logic)
* Arm B (test)   : contact the 30% with the highest predicted *uplift*
* Same capacity in both arms, so the comparison isolates targeting quality.
* Primary metric : 60-day order conversion, intent-to-treat over ALL accounts in arm
* Guardrails     : unsubscribe / opt-out rate, complaint rate, rep workload
* Decision rule  : ship if lift > 0 with p < 0.05 AND guardrails within +0.3 pts
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from ..config import FIG_ML, METRICS, NEUTRAL, PALETTE, SEED, TEXT_2
from ..data.generate_ml_data import generate
from ..viz import save
from .features import feature_frame

CAPACITY = 0.30
VALUE_PER_CONVERSION = 3_500   # blended incremental gross margin per converted account (assumption)
ALPHA, POWER = 0.05, 0.80


def required_n(p_base: float, mde_pts: float) -> int:
    es = proportion_effectsize(p_base + mde_pts, p_base)
    return int(np.ceil(NormalIndPower().solve_power(effect_size=es, alpha=ALPHA, power=POWER, ratio=1.0)))


def run(uplift_model=None, n_pilot: int = 20_000) -> dict:
    import joblib
    from ..config import MODELS
    model = uplift_model or joblib.load(MODELS / "nba_uplift_tlearner.joblib")

    pop = generate(n=n_pilot, seed=SEED + 2027)          # fresh, unseen population
    rng = np.random.default_rng(SEED + 1)
    X = feature_frame(pop)
    arm = np.where(rng.random(len(pop)) < 0.5, "A", "B")
    k_share = CAPACITY

    upl = model.predict_uplift(X)
    resp = model.m1.predict_proba(X)[:, 1]                       # response-model challenger (offline only)
    bau = pop.digital_engagement_score.to_numpy() + rng.normal(0, 0.01, len(pop))

    # ----- offline policy evaluation (IPW on the population's randomized history)
    t_hist, y_hist = pop.nba_outreach.to_numpy(), pop.converted_60d.to_numpy()
    def ipw_value(contact_mask):
        match = (t_hist == contact_mask.astype(int))
        return float(np.mean(y_hist * match / 0.5))
    none = np.zeros(len(pop), dtype=bool)
    v_none = ipw_value(none)
    policies = {"Random 30%": rng.random(len(pop)), "BAU (engaged)": bau,
                "Response model": resp, "Uplift model": upl}
    ope = {}
    for name, sc in policies.items():
        mask = sc >= np.quantile(sc, 1 - k_share)
        ope[name] = {"ipw_incremental_per_1000": (ipw_value(mask) - v_none) * 1000,
                     "true_incremental_per_1000": float(pop.true_uplift.to_numpy()[mask].sum() / len(pop) * 1000)}

    contact = np.zeros(len(pop), dtype=bool)
    for a, score in (("A", bau), ("B", upl)):
        m = arm == a
        cutoff = np.quantile(score[m], 1 - k_share)
        contact[m] = score[m] >= cutoff

    p = (pop.true_p_control + contact * pop.true_uplift).clip(0, 1).to_numpy()
    conv = rng.random(len(pop)) < p
    unsub = rng.random(len(pop)) < (0.010 + contact * (0.004 + 0.015 * (pop.email_open_rate < 0.15)))

    res = {}
    for a in ("A", "B"):
        m = arm == a
        res[a] = {"n": int(m.sum()), "conversions": int(conv[m].sum()), "rate": float(conv[m].mean()),
                  "unsub_rate": float(unsub[m].mean()), "contacted": int(contact[m].sum())}
    diff = res["B"]["rate"] - res["A"]["rate"]
    se = np.sqrt(res["A"]["rate"] * (1 - res["A"]["rate"]) / res["A"]["n"] + res["B"]["rate"] * (1 - res["B"]["rate"]) / res["B"]["n"])
    z, pval = proportions_ztest([res["B"]["conversions"], res["A"]["conversions"]], [res["B"]["n"], res["A"]["n"]],
                                alternative="larger")
    guard_diff = res["B"]["unsub_rate"] - res["A"]["unsub_rate"]
    incr_per_1000 = diff * 1000
    out = {
        "design": {"capacity": CAPACITY, "alpha": ALPHA, "power": POWER, "unit": "account",
                   "primary_metric": "60-day order conversion (ITT)", "guardrail": "unsubscribe rate <= +0.3 pts"},
        "power_analysis": {"baseline_rate": res["A"]["rate"],
                           "n_per_arm_for_1pt_mde": required_n(res["A"]["rate"], 0.010),
                           "n_per_arm_for_1_5pt_mde": required_n(res["A"]["rate"], 0.015),
                           "n_per_arm_for_2pt_mde": required_n(res["A"]["rate"], 0.020)},
        "offline_policy_evaluation": ope,
        "arms": res,
        "abs_lift_pts": float(diff * 100), "ci95_pts": [float((diff - 1.96 * se) * 100), float((diff + 1.96 * se) * 100)],
        "relative_lift": float(diff / res["A"]["rate"]), "p_value": float(pval), "z": float(z),
        "guardrail_unsub_diff_pts": float(guard_diff * 100),
        "incremental_conversions_per_1000_accounts": float(incr_per_1000),
        "value_per_conversion_assumed_usd": VALUE_PER_CONVERSION,
        "incremental_value_per_10k_accounts_usd": float(diff * 10_000 * VALUE_PER_CONVERSION),
    }
    out["decision"] = "SHIP / scale to Wave 2" if (pval < ALPHA and diff > 0 and guard_diff * 100 <= 0.3) else "ITERATE"

    # ----- figures
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    axes = axes.ravel()
    names = list(ope)
    ipw = [ope[n]["ipw_incremental_per_1000"] for n in names]
    tru = [ope[n]["true_incremental_per_1000"] for n in names]
    xi = np.arange(len(names))
    axes[0].bar(xi - 0.19, ipw, width=0.38, color=PALETTE[0], label="IPW estimate (from randomized history)")
    axes[0].bar(xi + 0.19, tru, width=0.38, color=PALETTE[2], label="Synthetic ground truth")
    axes[0].set_xticks(xi, names)
    axes[0].set(title="1. Offline policy evaluation (pre-pilot)", ylabel="Incremental conversions / 1,000 accounts")
    axes[0].legend(loc="upper left")

    mdes = np.linspace(0.006, 0.03, 40)
    ns = [required_n(res["A"]["rate"], m) for m in mdes]
    axes[1].plot(mdes * 100, np.array(ns) / 1000, color=PALETTE[0])
    n_ = required_n(res["A"]["rate"], 0.015)
    axes[1].scatter([1.5], [n_ / 1000], color=PALETTE[1], s=40, zorder=3)
    axes[1].annotate(f"MDE 1.5 pts needs {n_:,} per arm", (1.5, n_ / 1000), xytext=(12, 10),
                     textcoords="offset points", color=TEXT_2, fontsize=9)
    axes[1].set(title="2. Power analysis (alpha 0.05, power 0.80)", xlabel="Minimum detectable effect (pts)",
                ylabel="Accounts per arm (thousands)")

    rates = [res["A"]["rate"] * 100, res["B"]["rate"] * 100]
    errs = [1.96 * np.sqrt(r / 100 * (1 - r / 100) / res[a]["n"]) * 100 for r, a in zip(rates, "AB")]
    axes[2].bar(["A: BAU (most engaged)", "B: Uplift-guided"], rates, yerr=errs, capsize=6,
                color=[NEUTRAL, PALETTE[0]], width=0.55, error_kw={"ecolor": TEXT_2, "lw": 1})
    for i, r in enumerate(rates):
        axes[2].text(i, r + errs[i] + 0.3, f"{r:.1f}%", ha="center", color=TEXT_2)
    axes[2].set(title=f"3. Primary metric: 60-day conversion (ITT)\nLift +{diff*100:.2f} pts, 95% CI "
                      f"[{out['ci95_pts'][0]:.2f}, {out['ci95_pts'][1]:.2f}], p={pval:.2g}",
                ylabel="Conversion rate (%)", ylim=(0, max(rates) * 1.3))
    ur = [res["A"]["unsub_rate"] * 100, res["B"]["unsub_rate"] * 100]
    axes[3].bar(["A: BAU", "B: Uplift"], ur, color=[NEUTRAL, PALETTE[0]], width=0.5)
    axes[3].axhline(ur[0] + 0.3, color=PALETTE[7], ls="--", lw=1.2, label="Guardrail limit (A + 0.3 pts)")
    for i, r in enumerate(ur):
        axes[3].text(i, r + 0.05, f"{r:.2f}%", ha="center", color=TEXT_2)
    axes[3].set(title="4. Guardrail: unsubscribe rate", ylabel="Rate (%)", ylim=(0, max(ur) * 1.8))
    axes[3].legend(loc="upper left")
    save(fig, FIG_ML / "07_pilot_power_and_readout.png",
         f"Simulated pilot on an unseen synthetic population (n={n_pilot:,}); both arms contact 30% of accounts. Decision: {out['decision']}.")

    (METRICS / "pilot_ab_test.json").write_text(json.dumps(out, indent=2))
    print(f"[pilot] A={res['A']['rate']:.3%} B={res['B']['rate']:.3%} lift={diff*100:.2f}pts p={pval:.2g} -> {out['decision']}")
    return out


if __name__ == "__main__":
    run()
