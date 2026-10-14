"""Synthetic analytics warehouse (star schema) for marketing-mix and CX KPIs.

Grain and tables
----------------
Dimensions : dim_date (week), dim_brand, dim_channel, dim_region, dim_campaign,
             dim_account, dim_touchpoint
Facts      : fact_marketing_activity  week x brand x region x channel (x campaign)
             fact_orders              week x brand x region
             fact_cx_interaction      one row per customer interaction

Orders are generated from a *known* media response model (geometric adstock +
Hill saturation, channel-level ROI targets). The truth is written to
``data/processed/mmm_ground_truth.json`` so the in-house MMM can be validated
against it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import BRANDS, CHANNELS, DATA_PROCESSED, DATA_WAREHOUSE, REGIONS, SEED

N_WEEKS = 156
START = "2023-01-02"
REGION_W = dict(zip(REGIONS, [0.25, 0.22, 0.19, 0.14, 0.20]))

# Mean weekly national spend (USD) per brand x channel
SPEND = {
    "BEC_IA": dict(REP=120e3, EML=8e3, SRCH=15e3, DISP=12e3, SOC=10e3, CONF=30e3, WEB=6e3),
    "BEC_HEM": dict(REP=85e3, EML=6e3, SRCH=10e3, DISP=9e3, SOC=7e3, CONF=20e3, WEB=4e3),
    "C2N_AD2": dict(REP=60e3, EML=5e3, SRCH=18e3, DISP=10e3, SOC=12e3, CONF=15e3, WEB=6e3),
}
# True average ROI (incremental revenue / spend)
TRUE_ROI = {
    "BEC_IA": dict(REP=1.9, EML=4.5, SRCH=2.6, DISP=0.9, SOC=1.2, CONF=1.4, WEB=3.0),
    "BEC_HEM": dict(REP=1.6, EML=3.8, SRCH=2.2, DISP=0.7, SOC=1.0, CONF=1.1, WEB=2.4),
    "C2N_AD2": dict(REP=2.4, EML=5.2, SRCH=3.4, DISP=1.1, SOC=1.8, CONF=0.9, WEB=3.6),
}
DECAY = dict(REP=0.55, EML=0.20, SRCH=0.10, DISP=0.30, SOC=0.30, CONF=0.60, WEB=0.40)
HILL_S = dict(REP=1.0, EML=1.0, SRCH=1.3, DISP=1.3, SOC=1.2, CONF=1.0, WEB=1.0)
BASE_ORDERS = {"BEC_IA": 1500.0, "BEC_HEM": 1750.0, "C2N_AD2": 520.0}   # weekly national baseline
TREND = {"BEC_IA": 0.04, "BEC_HEM": 0.02, "C2N_AD2": 0.38}           # annual growth of base


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def hill(x: np.ndarray, k: float, s: float) -> np.ndarray:
    x = np.maximum(x, 0)
    return x ** s / (x ** s + k ** s)


def _activity_metrics(ch: str, units: float) -> tuple[int, int, int]:
    """Return (impressions, clicks, engagements) given purchased units for a channel."""
    if ch == "SRCH":       # units = clicks
        return int(units * 25), int(units), int(units)
    if ch in ("DISP", "SOC"):  # units = impressions
        clicks = int(units * 0.012)
        return int(units), clicks, clicks
    if ch == "EML":        # units = sends
        return int(units), int(units * 0.03), int(units * 0.22)
    return 0, 0, int(units)  # REP calls, CONF booth leads, WEB attendees


def _dim_date() -> pd.DataFrame:
    wk = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    d = pd.DataFrame({"week_start": wk})
    d["date_key"] = d.week_start.dt.strftime("%Y%m%d").astype(int)
    d["year"] = d.week_start.dt.year
    d["quarter"] = d.week_start.dt.quarter
    d["month"] = d.week_start.dt.month
    d["iso_week"] = d.week_start.dt.isocalendar().week.astype(int)
    d["year_quarter"] = d.year.astype(str) + "-Q" + d.quarter.astype(str)
    d["is_holiday_week"] = d.iso_week.isin([1, 22, 27, 36, 47, 52]).astype(int)
    # congress calendar: ADLM (lab, late Jul), AAN (Apr), AAIC (late Jul), CTAD (Dec)
    d["congress"] = np.select(
        [d.iso_week.isin([30, 31]), d.iso_week.isin([15, 16]), d.iso_week.isin([48, 49])],
        ["ADLM / AAIC", "AAN", "CTAD"], default="")
    return d[["date_key", "week_start", "year", "quarter", "month", "iso_week", "year_quarter",
              "is_holiday_week", "congress"]]


def _spend_series(rng, brand: str, ch: str, dd: pd.DataFrame) -> np.ndarray:
    mean = SPEND[brand][ch]
    t = np.arange(N_WEEKS)
    growth = 1.08 ** (t / 52)
    # each channel has its own planning rhythm (phase) so media is not collinear with seasonality
    season = 1 + 0.15 * np.sin(2 * np.pi * (t - rng.integers(0, 52)) / 26)
    noise = rng.lognormal(0, 0.35, N_WEEKS)
    if ch == "CONF":
        is_c2n = brand.startswith("C2N")
        cong = dd.congress.to_numpy()
        spike = np.where(cong == "ADLM / AAIC", 1.0, 0.0)
        spike += np.where(cong == ("CTAD" if is_c2n else "AAN"), 0.55, 0.0)
        spike += np.where(cong == ("AAN" if is_c2n else "CTAD"), 0.35 if is_c2n else 0.0, 0.0)
        base = 0.08 + spike
        x = base / base.mean() * mean * noise
    elif ch in ("DISP", "SOC"):
        flight = (rng.random(N_WEEKS) < 0.65).astype(float)
        flight = np.convolve(flight, np.ones(3) / 3, mode="same").round()   # contiguous bursts
        x = mean * flight / max(flight.mean(), 0.1) * noise * season
    elif ch == "REP":
        dips = np.where(dd.is_holiday_week.to_numpy() == 1, 0.55, 1.0)
        vacancy = 1 - 0.25 * (np.convolve(rng.random(N_WEEKS) < 0.08, np.ones(6), mode="same") > 0)
        x = mean * season * dips * vacancy * rng.lognormal(0, 0.10, N_WEEKS)
    else:
        x = mean * season * noise
    return x * growth / growth.mean()


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dd = _dim_date()
    t = np.arange(N_WEEKS)

    dim_brand = pd.DataFrame([
        {"brand_key": i + 1, "brand_code": c, "brand_name": v[0], "company": v[1],
         "clinical_area": v[2], "avg_order_value_usd": v[3]}
        for i, (c, v) in enumerate(BRANDS.items())])
    dim_channel = pd.DataFrame([
        {"channel_key": i + 1, "channel_code": c, "channel_name": v[0], "promotion_type": v[1],
         "channel_group": v[2], "true_decay_hidden": DECAY[c]}
        for i, (c, v) in enumerate(CHANNELS.items())]).drop(columns="true_decay_hidden")
    dim_region = pd.DataFrame({"region_key": range(1, 6), "region_name": REGIONS,
                               "territories": [14, 12, 11, 8, 12], "spend_weight": list(REGION_W.values())})

    bkey = dict(zip(dim_brand.brand_code, dim_brand.brand_key))
    ckey = dict(zip(dim_channel.channel_code, dim_channel.channel_key))
    rkey = dict(zip(dim_region.region_name, dim_region.region_key))

    objectives = dict(REP="Conversion", EML="Retention", SRCH="Conversion", DISP="Awareness",
                      SOC="Consideration", CONF="Awareness", WEB="Consideration")
    audiences = {"BEC_IA": "Lab Directors & Procurement", "BEC_HEM": "Lab Directors & Procurement",
                 "C2N_AD2": "Neurologists & PCPs"}
    camps = []
    for b in BRANDS:
        for c in CHANNELS:
            for y in (2023, 2024, 2025):
                camps.append({"campaign_key": len(camps) + 1,
                              "campaign_name": f"{b}-{y}-{c}-{objectives[c]}",
                              "brand_key": bkey[b], "channel_key": ckey[c], "fiscal_year": y,
                              "objective": objectives[c], "target_audience": audiences[b],
                              "mlr_approval_id": f"MLR-{b[:3]}-{y}-{rng.integers(1000, 9999)}",
                              "start_date": f"{y}-01-01", "end_date": f"{y}-12-31"})
    dim_campaign = pd.DataFrame(camps)
    camp_lookup = {(r.brand_key, r.channel_key, r.fiscal_year): r.campaign_key for r in dim_campaign.itertuples()}

    unit_cost = dict(REP=230, EML=0.09, SRCH=6.5, DISP=0.028, SOC=0.048, CONF=180, WEB=45)  # per engagement/impr
    fact_mkt, fact_ord, truth = [], [], {}
    for b in BRANDS:
        aov = BRANDS[b][3]
        truth[b] = {"aov": aov, "channels": {}}
        reg_media = {r: np.zeros(N_WEEKS) for r in REGIONS}
        for c in CHANNELS:
            x_nat = _spend_series(rng, b, c, dd)
            k = adstock(x_nat, DECAY[c]).mean()
            h_nat = hill(adstock(x_nat, DECAY[c]), k, HILL_S[c])
            beta = TRUE_ROI[b][c] * x_nat.mean() / aov / h_nat.mean()   # orders at saturation
            truth[b]["channels"][c] = {"decay": DECAY[c], "hill_k": float(k), "hill_s": HILL_S[c],
                                       "beta_orders": float(beta), "true_roi": TRUE_ROI[b][c],
                                       "mean_weekly_spend": float(x_nat.mean())}
            for r in REGIONS:
                w = REGION_W[r] * rng.lognormal(0, 0.05)
                x_r = x_nat * w * rng.lognormal(0, 0.30, N_WEEKS)   # geo-level flighting variation
                contrib = beta * REGION_W[r] * hill(adstock(x_r, DECAY[c]), k * REGION_W[r], HILL_S[c])
                reg_media[r] += contrib
                yr = dd.year.to_numpy()
                for i in range(N_WEEKS):
                    sp = x_r[i]
                    impr, clicks, eng = _activity_metrics(c, sp / unit_cost[c])
                    fact_mkt.append((int(dd.date_key[i]), bkey[b], rkey[r], ckey[c],
                                     camp_lookup[(bkey[b], ckey[c], int(yr[i]))], round(sp, 2),
                                     impr, clicks, eng))

        season = 1 + 0.06 * np.sin(2 * np.pi * (t - 10) / 52) - 0.10 * dd.is_holiday_week.to_numpy()
        base_nat = BASE_ORDERS[b] * (1 + TREND[b]) ** (t / 52) * season
        truth[b]["base_mean_weekly_orders"] = float(base_nat.mean())
        for r in REGIONS:
            lam = base_nat * REGION_W[r] + reg_media[r]
            orders = rng.poisson(lam * rng.lognormal(0, 0.02, N_WEEKS))
            for i in range(N_WEEKS):
                fact_ord.append((int(dd.date_key[i]), bkey[b], rkey[r], int(orders[i]),
                                 round(orders[i] * aov * rng.normal(1, 0.03), 2),
                                 int(rng.poisson(max(orders[i] * (0.18 if b.startswith("C2N") else 0.06), 0.1)))))

    fact_marketing_activity = pd.DataFrame(fact_mkt, columns=[
        "date_key", "brand_key", "region_key", "channel_key", "campaign_key", "spend_usd",
        "impressions", "clicks", "engagements"])
    fact_orders = pd.DataFrame(fact_ord, columns=[
        "date_key", "brand_key", "region_key", "orders", "revenue_usd", "new_accounts"])

    # ---------------- accounts & CX interactions
    n_acc = 4000
    acc_brand = rng.choice(list(BRANDS), size=n_acc, p=[0.30, 0.25, 0.45])
    ctype = np.where(np.char.startswith(acc_brand.astype(str), "C2N"),
                     rng.choice(["Neurology Practice", "Memory Clinic", "Primary Care", "Geriatric Psychiatry"], n_acc),
                     rng.choice(["Hospital Lab", "Reference Lab", "IDN Core Lab", "Physician Office Lab"], n_acc))
    dim_account = pd.DataFrame({
        "account_key": range(1, n_acc + 1),
        "account_id": [f"ACC-{i:05d}" for i in range(n_acc)],
        "primary_brand_key": [bkey[b] for b in acc_brand],
        "customer_type": ctype,
        "region_key": rng.choice(range(1, 6), n_acc, p=list(REGION_W.values())),
        "account_tier": rng.choice(["A", "B", "C"], n_acc, p=[0.2, 0.35, 0.45]),
        "value_decile": rng.integers(1, 11, n_acc),
        "consent_email": (rng.random(n_acc) < 0.82).astype(int),
    })
    touch = ["Tech Support Ticket", "Field Service Visit", "Customer Portal", "Rep Visit",
             "Peer Webinar", "Email Click", "Results / Billing Inquiry"]
    dim_touchpoint = pd.DataFrame({"touchpoint_key": range(1, len(touch) + 1), "touchpoint_name": touch,
                                   "is_service": [1, 1, 0, 0, 0, 0, 1], "is_digital": [0, 0, 1, 0, 1, 1, 0]})
    n_int = 60_000
    wk_idx = rng.integers(0, N_WEEKS, n_int)
    program = np.clip((wk_idx - 78) / 26, 0, 1)   # CX program launched mid-2024, ramps over 2 quarters
    tp = rng.choice(dim_touchpoint.touchpoint_key, n_int, p=[0.18, 0.08, 0.26, 0.14, 0.08, 0.18, 0.08])
    is_service = np.isin(tp, [1, 2, 7])
    res_hrs = np.where(is_service, rng.lognormal(3.0 - 0.25 * program, 0.6), np.nan)
    fcr = np.where(is_service, (rng.random(n_int) < 0.62 + 0.10 * program).astype(float), np.nan)
    csat = np.clip(np.round(rng.normal(3.75 + 0.30 * program - 0.004 * np.nan_to_num(res_hrs), 0.8)), 1, 5)
    ces = np.clip(np.round(rng.normal(3.4 - 0.4 * program, 1.1)), 1, 7)
    nps_resp = rng.random(n_int) < 0.12
    nps = np.where(nps_resp, np.clip(np.round(rng.normal(7.9 + 0.7 * program, 1.7)), 0, 10), np.nan)
    acc = rng.integers(1, n_acc + 1, n_int)
    fact_cx_interaction = pd.DataFrame({
        "interaction_id": np.arange(1, n_int + 1),
        "date_key": dd.date_key.to_numpy()[wk_idx],
        "account_key": acc,
        "brand_key": dim_account.primary_brand_key.to_numpy()[acc - 1],
        "touchpoint_key": tp,
        "resolution_hrs": np.round(res_hrs, 1),
        "first_contact_resolved": fcr,
        "csat_1to5": csat,
        "customer_effort_1to7": ces,
        "nps_0to10": nps,
    }).sort_values(["date_key", "interaction_id"]).reset_index(drop=True)

    return {
        "dim_date": dd, "dim_brand": dim_brand, "dim_channel": dim_channel, "dim_region": dim_region,
        "dim_campaign": dim_campaign, "dim_account": dim_account, "dim_touchpoint": dim_touchpoint,
        "fact_marketing_activity": fact_marketing_activity, "fact_orders": fact_orders,
        "fact_cx_interaction": fact_cx_interaction, "_truth": truth,
    }


def main() -> dict[str, pd.DataFrame]:
    tables = generate()
    truth = tables.pop("_truth")
    for name, df in tables.items():
        df.to_csv(DATA_WAREHOUSE / f"{name}.csv", index=False)
    (DATA_PROCESSED / "mmm_ground_truth.json").write_text(json.dumps(truth, indent=2))
    print("[analytics-data] " + ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))
    return tables


if __name__ == "__main__":
    main()
