"""Load the star schema, enforce data contracts, and draw the ER diagram.

Layering (mirrors a Snowflake / Redshift + Power BI deployment)
---------------------------------------------------------------
1. Data layer      : conformed dimensions + fact tables (data/warehouse, sql/01_ddl.sql)
2. Semantic layer  : governed KPI measures (sql/02_kpi_views.sql, semantic_model/*.dax)
3. Visualization   : Power BI report / PNG outputs (reports/figures/analytics)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from ..config import DATA_WAREHOUSE, FIG_ANALYTICS, NEUTRAL, PALETTE, SURFACE, TEXT, TEXT_2
from ..viz import save

TABLES = ["dim_date", "dim_brand", "dim_channel", "dim_region", "dim_campaign", "dim_account",
          "dim_touchpoint", "fact_marketing_activity", "fact_orders", "fact_cx_interaction"]

# Data contracts: primary keys, grain, and foreign keys that must resolve.
CONTRACTS = {
    "fact_marketing_activity": {
        "grain": ["date_key", "brand_key", "region_key", "channel_key"],
        "fks": {"date_key": "dim_date", "brand_key": "dim_brand", "region_key": "dim_region",
                "channel_key": "dim_channel", "campaign_key": "dim_campaign"},
        "non_negative": ["spend_usd", "impressions", "clicks", "engagements"],
    },
    "fact_orders": {
        "grain": ["date_key", "brand_key", "region_key"],
        "fks": {"date_key": "dim_date", "brand_key": "dim_brand", "region_key": "dim_region"},
        "non_negative": ["orders", "revenue_usd", "new_accounts"],
    },
    "fact_cx_interaction": {
        "grain": ["interaction_id"],
        "fks": {"date_key": "dim_date", "account_key": "dim_account", "brand_key": "dim_brand",
                "touchpoint_key": "dim_touchpoint"},
        "non_negative": ["resolution_hrs"],
    },
}


def load() -> dict[str, pd.DataFrame]:
    t = {name: pd.read_csv(DATA_WAREHOUSE / f"{name}.csv") for name in TABLES}
    t["dim_date"]["week_start"] = pd.to_datetime(t["dim_date"]["week_start"])
    return t


def validate(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quality gate run in CI before any model or report refresh."""
    rows = []
    for fact, c in CONTRACTS.items():
        df = t[fact]
        rows.append({"table": fact, "check": "grain unique", "passed": not df.duplicated(c["grain"]).any()})
        for fk, dim in c["fks"].items():
            ok = df[fk].isin(t[dim][fk]).all()
            rows.append({"table": fact, "check": f"FK {fk} -> {dim}", "passed": bool(ok)})
        for col in c["non_negative"]:
            rows.append({"table": fact, "check": f"{col} >= 0", "passed": bool((df[col].dropna() >= 0).all())})
    return pd.DataFrame(rows)


def model_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MMM input at geo grain: brand x region x week, one spend column per channel,
    joined to orders. Equivalent to sql/03_mmm_model_frame.sql."""
    keys = ["brand_code", "region_key", "date_key"]
    mkt = (t["fact_marketing_activity"]
           .merge(t["dim_channel"][["channel_key", "channel_code"]], on="channel_key")
           .merge(t["dim_brand"][["brand_key", "brand_code"]], on="brand_key")
           .pivot_table(index=keys, columns="channel_code", values="spend_usd", aggfunc="sum")
           .reset_index())
    orders = (t["fact_orders"].merge(t["dim_brand"][["brand_key", "brand_code"]], on="brand_key")
              .groupby(keys, as_index=False)[["orders", "revenue_usd"]].sum())
    mf = mkt.merge(orders, on=keys).merge(
        t["dim_date"][["date_key", "week_start", "is_holiday_week", "iso_week"]], on="date_key")
    return mf.sort_values(["brand_code", "region_key", "week_start"]).reset_index(drop=True)


def draw_er_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, title, cols, color):
        h = 0.36 + 0.24 * len(cols)
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=SURFACE, ec=color, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, y - 0.36), w, 0.36, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec=color, lw=1.6))
        ax.text(x + 0.12, y - 0.18, title, color="white", fontsize=9.5, weight="bold", va="center")
        for i, c in enumerate(cols):
            ax.text(x + 0.12, y - 0.52 - 0.24 * i, c, color=TEXT if c.startswith(("PK", "FK")) else TEXT_2,
                    fontsize=8, va="center")
        return (x, y, w, h)

    fact_c, dim_c = PALETTE[0], PALETTE[6]
    f1 = box(4.6, 6.9, 3.6, "fact_marketing_activity", ["grain: week x brand x region x channel",
              "FK date_key, brand_key, region_key", "FK channel_key, campaign_key", "spend_usd, impressions",
              "clicks, engagements"], fact_c)
    f2 = box(4.6, 4.45, 3.6, "fact_orders", ["grain: week x brand x region", "FK date_key, brand_key, region_key",
              "orders, revenue_usd, new_accounts"], fact_c)
    f3 = box(4.6, 2.55, 3.6, "fact_cx_interaction", ["grain: one customer interaction",
              "FK date_key, account_key, brand_key", "FK touchpoint_key", "csat, effort, nps, resolution_hrs, FCR"], fact_c)
    dims_l = [(0.3, 6.9, "dim_date", ["PK date_key", "week_start, year, quarter", "congress, holiday flags"]),
              (0.3, 4.95, "dim_brand", ["PK brand_key", "brand, company", "clinical_area, AOV"]),
              (0.3, 3.0, "dim_region", ["PK region_key", "region_name, territories"])]
    dims_r = [(9.1, 6.9, "dim_channel", ["PK channel_key", "channel, promotion_type", "channel_group"]),
              (9.1, 5.15, "dim_campaign", ["PK campaign_key", "objective, audience", "MLR approval id"]),
              (9.1, 3.4, "dim_account", ["PK account_key", "customer_type, tier", "value_decile, consent"]),
              (9.1, 1.65, "dim_touchpoint", ["PK touchpoint_key", "touchpoint, is_service", "is_digital"])]
    for x, y, n, c in dims_l + dims_r:
        b = box(x, y, 3.3 if x > 9 else 3.4, n, c, dim_c)
        cx = b[0] + b[2] if x < 4 else b[0]
        cy = b[1] - b[3] / 2
        targets = {"dim_date": [f1, f2, f3], "dim_brand": [f1, f2, f3], "dim_region": [f1, f2],
                   "dim_channel": [f1], "dim_campaign": [f1], "dim_account": [f3], "dim_touchpoint": [f3]}[n]
        for f in targets:
            fx = f[0] if x < 4 else f[0] + f[2]
            fy = f[1] - f[3] / 2
            ax.plot([cx, fx], [cy, fy], color=NEUTRAL, lw=1.0, alpha=0.7, zorder=0)
    ax.text(0.3, 0.35, "Conformed dimensions are shared across facts, so marketing spend, orders and CX signals "
            "slice by the same brand, region and week.", fontsize=8.5, color=TEXT_2)
    ax.set_title("CX analytics star schema (data layer)", loc="left")
    save(fig, FIG_ANALYTICS / "01_star_schema_er_diagram.png")
