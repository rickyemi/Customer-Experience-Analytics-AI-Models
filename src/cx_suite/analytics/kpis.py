"""Governed KPI framework: one catalog, one calculation path, many consumers.

The catalog below is the single source of truth. It is exported to
``semantic_model/kpi_catalog.csv`` (business glossary), mirrored by the SQL
views in ``sql/02_kpi_views.sql`` and the DAX measures in
``semantic_model/measures.dax`` so Python, SQL and Power BI agree.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, FIG_ANALYTICS, NEUTRAL, PALETTE, ROOT, SURFACE, TEXT, TEXT_2, GRID
from ..viz import save

KPI_CATALOG = [
    # tier, name, definition, formula, grain, direction, target, owner
    ("North Star", "Incremental Marketing ROI", "Incremental revenue attributed by the MMM per $1 of marketing spend",
     "SUM(mmm_incremental_revenue) / SUM(spend_usd)", "brand x channel x quarter", "up", ">= 2.0", "Performance Optimization"),
    ("North Star", "Net Promoter Score (NPS)", "% promoters (9-10) minus % detractors (0-6) among survey responses",
     "100 * (COUNT(nps>=9) - COUNT(nps<=6)) / COUNT(nps)", "brand x quarter", "up", ">= 40", "Customer Insights"),
    ("Driver", "Cost per Order (CPO)", "Total marketing spend divided by orders",
     "SUM(spend_usd) / SUM(orders)", "brand x region x quarter", "down", "brand-specific", "Performance Optimization"),
    ("Driver", "Revenue per Marketing $", "Total (not incremental) revenue per $ spend; descriptive only",
     "SUM(revenue_usd) / SUM(spend_usd)", "brand x quarter", "up", "trend", "Performance Optimization"),
    ("Driver", "New Account Acquisition", "Accounts placing a first order in the period",
     "SUM(new_accounts)", "brand x region x quarter", "up", "+10% YoY", "Commercial Ops"),
    ("Driver", "Digital Engagement Rate", "Clicks per impression across paid + owned digital",
     "SUM(clicks) / SUM(impressions)", "channel x quarter", "up", ">= 1.2%", "Digital Analytics"),
    ("Driver", "Non-personal Promotion Share", "Share of spend in non-personal channels",
     "SUM(spend where promotion_type='Non-personal') / SUM(spend)", "brand x quarter", "context", "35-45%", "Brand Ops"),
    ("Diagnostic", "CSAT Top-2-Box", "% of interactions rated 4 or 5 of 5",
     "AVG(csat_1to5 >= 4)", "touchpoint x quarter", "up", ">= 75%", "Customer Service"),
    ("Diagnostic", "First Contact Resolution", "% of service interactions resolved on first contact",
     "AVG(first_contact_resolved) WHERE is_service", "touchpoint x quarter", "up", ">= 70%", "Customer Service"),
    ("Diagnostic", "Avg Resolution Hours", "Mean hours from case open to resolution (service touchpoints)",
     "AVG(resolution_hrs) WHERE is_service", "touchpoint x quarter", "down", "<= 18 h", "Customer Service"),
    ("Diagnostic", "Customer Effort Score", "Mean effort score, 1 (low effort) to 7 (high effort)",
     "AVG(customer_effort_1to7)", "touchpoint x quarter", "down", "<= 3.0", "Customer Insights"),
    ("Diagnostic", "Active Accounts", "Distinct accounts with any interaction in the period",
     "COUNT(DISTINCT account_key)", "brand x quarter", "up", "trend", "Commercial Ops"),
]


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(KPI_CATALOG, columns=["tier", "kpi", "definition", "formula", "grain", "direction",
                                              "target", "owner"])


def quarterly_kpis(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = t["dim_date"][["date_key", "year_quarter", "year"]]
    mkt = t["fact_marketing_activity"].merge(d, on="date_key").merge(
        t["dim_channel"][["channel_key", "promotion_type"]], on="channel_key")
    ords = t["fact_orders"].merge(d, on="date_key")
    cx = t["fact_cx_interaction"].merge(d, on="date_key").merge(
        t["dim_touchpoint"][["touchpoint_key", "is_service"]], on="touchpoint_key")

    m = mkt.groupby("year_quarter").agg(spend=("spend_usd", "sum"), clicks=("clicks", "sum"), impr=("impressions", "sum"))
    m["np_spend"] = mkt[mkt.promotion_type == "Non-personal"].groupby("year_quarter").spend_usd.sum()
    o = ords.groupby("year_quarter").agg(orders=("orders", "sum"), revenue=("revenue_usd", "sum"),
                                         new_accounts=("new_accounts", "sum"))
    svc = cx[cx.is_service == 1]

    def nps(s):
        s = s.dropna()
        return 100 * ((s >= 9).mean() - (s <= 6).mean())

    c = pd.DataFrame({
        "nps": cx.groupby("year_quarter").nps_0to10.apply(nps),
        "csat_top2": cx.groupby("year_quarter").csat_1to5.apply(lambda s: (s >= 4).mean()),
        "fcr": svc.groupby("year_quarter").first_contact_resolved.mean(),
        "resolution_hrs": svc.groupby("year_quarter").resolution_hrs.mean(),
        "effort": cx.groupby("year_quarter").customer_effort_1to7.mean(),
        "active_accounts": cx.groupby("year_quarter").account_key.nunique(),
    })
    k = m.join(o).join(c)
    k["cpo"] = k.spend / k.orders
    k["rev_per_mkt_dollar"] = k.revenue / k.spend
    k["digital_engagement_rate"] = k.clicks / k.impr
    k["non_personal_share"] = k.np_spend / k.spend
    return k.reset_index()


def brand_quarterly(t) -> pd.DataFrame:
    d = t["dim_date"][["date_key", "year_quarter"]]
    b = t["dim_brand"][["brand_key", "brand_name"]]
    mkt = t["fact_marketing_activity"].merge(d, on="date_key").merge(b, on="brand_key")
    ords = t["fact_orders"].merge(d, on="date_key").merge(b, on="brand_key")
    s = mkt.groupby(["brand_name", "year_quarter"]).spend_usd.sum()
    o = ords.groupby(["brand_name", "year_quarter"])[["orders", "revenue_usd"]].sum()
    out = o.join(s).reset_index()
    out["cpo"] = out.spend_usd / out.orders
    return out


def _yoy(k: pd.DataFrame, col: str, agg="mean"):
    k = k.assign(year=k.year_quarter.str[:4].astype(int))
    g = k.groupby("year")[col].agg(agg)
    return g.get(2025, np.nan), g.get(2024, np.nan)


def figures(t, k: pd.DataFrame, mmm_roi: float | None = None) -> None:
    # ---- scorecard tiles (FY2025 vs FY2024)
    kk = k.assign(year=k.year_quarter.str[:4].astype(int))
    fy = kk.groupby("year").agg(spend=("spend", "sum"), orders=("orders", "sum"), revenue=("revenue", "sum"),
                                new_accounts=("new_accounts", "sum"), nps=("nps", "mean"), csat=("csat_top2", "mean"),
                                fcr=("fcr", "mean"), res=("resolution_hrs", "mean"), effort=("effort", "mean"),
                                clicks=("clicks", "sum"), impr=("impr", "sum"))
    fy["cpo"] = fy.spend / fy.orders
    fy["der"] = fy.clicks / fy.impr
    tiles = [
        ("Incremental mktg ROI (MMM)", f"{mmm_roi:.2f}x" if mmm_roi else "n/a", "", None, "up"),
        ("Revenue (FY25)", f"${fy.revenue[2025]/1e6:,.0f}M", fy.revenue[2025] / fy.revenue[2024] - 1, None, "up"),
        ("Cost per order", f"${fy.cpo[2025]:,.0f}", fy.cpo[2025] / fy.cpo[2024] - 1, None, "down"),
        ("New accounts", f"{fy.new_accounts[2025]:,.0f}", fy.new_accounts[2025] / fy.new_accounts[2024] - 1, None, "up"),
        ("NPS", f"{fy.nps[2025]:.0f}", fy.nps[2025] - fy.nps[2024], "pts", "up"),
        ("CSAT top-2-box", f"{fy.csat[2025]:.0%}", (fy.csat[2025] - fy.csat[2024]) * 100, "pts", "up"),
        ("First contact resolution", f"{fy.fcr[2025]:.0%}", (fy.fcr[2025] - fy.fcr[2024]) * 100, "pts", "up"),
        ("Avg resolution hours", f"{fy.res[2025]:.1f} h", fy.res[2025] / fy.res[2024] - 1, None, "down"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 4.4))
    for ax, (name, val, delta, unit, good) in zip(axes.ravel(), tiles):
        ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.92, transform=ax.transAxes, fc=SURFACE, ec=GRID, lw=1.2))
        ax.text(0.08, 0.76, name, fontsize=9.5, color=TEXT_2, transform=ax.transAxes)
        ax.text(0.08, 0.40, val, fontsize=21, weight="bold", color=TEXT, transform=ax.transAxes)
        if delta != "":
            improved = (delta > 0) == (good == "up")
            arrow = "▲" if delta > 0 else "▼"
            txt = f"{arrow} {delta:+.1f} {unit} vs FY24" if unit else f"{arrow} {delta:+.1%} vs FY24"
            ax.text(0.08, 0.14, txt + ("  (better)" if improved else "  (worse)"), fontsize=8.5,
                    color=PALETTE[5] if improved else PALETTE[7], transform=ax.transAxes)
        else:
            ax.text(0.08, 0.14, "FY23-FY25, all brands & channels", fontsize=8.5, color=TEXT_2, transform=ax.transAxes)
    fig.suptitle("Governed KPI scorecard: FY2025", x=0.01, ha="left", fontsize=13, weight="bold")
    save(fig, FIG_ANALYTICS / "02_kpi_scorecard.png", "Synthetic data. Arrows show direction; 'better/worse' applies each KPI's desired direction.")

    # ---- CX KPI trends (small multiples, one measure each: no dual axes)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)
    specs = [("nps", "NPS", "{:.0f}"), ("csat_top2", "CSAT top-2-box", "{:.0%}"),
             ("fcr", "First contact resolution", "{:.0%}"), ("resolution_hrs", "Avg resolution hours", "{:.1f}")]
    x = np.arange(len(k))
    for ax, (col, title, fmt) in zip(axes.ravel(), specs):
        ax.plot(x, k[col], color=PALETTE[0], marker="o", ms=4)
        ax.axvline(6, color=NEUTRAL, ls="--", lw=1)
        ax.text(6.15, ax.get_ylim()[0] + 0.92 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "CX program launch",
                fontsize=8, color=TEXT_2)
        ax.set_title(title)
        ax.text(x[-1], k[col].iloc[-1], "  " + fmt.format(k[col].iloc[-1]), color=TEXT_2, fontsize=9, va="center")
        ax.set_xticks(x[::2], k.year_quarter[::2], rotation=0, fontsize=8)
    fig.suptitle("CX diagnostic KPIs by quarter", x=0.01, ha="left", fontsize=13, weight="bold")
    save(fig, FIG_ANALYTICS / "03_cx_kpi_trends.png")

    # ---- brand spend and orders (small multiples: two measures, two rows)
    bq = brand_quarterly(t)
    brands = list(bq.brand_name.unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    qs = sorted(bq.year_quarter.unique())
    xi = np.arange(len(qs))
    for i, b in enumerate(brands):
        s = bq[bq.brand_name == b].set_index("year_quarter").reindex(qs)
        axes[0].plot(xi, s.spend_usd / 1e6, color=PALETTE[i], label=b)
        axes[1].plot(xi, s.cpo, color=PALETTE[i], label=b)
    axes[0].set(title="Marketing spend by brand ($M per quarter)", ylabel="$M")
    axes[1].set(title="Cost per order by brand ($, log scale)", ylabel="$ per order", yscale="log")
    axes[0].legend(loc="upper left", ncol=3)
    axes[1].set_xticks(xi, qs, fontsize=8)
    save(fig, FIG_ANALYTICS / "04_brand_spend_cpo.png")


def export_catalog() -> pd.DataFrame:
    cat = catalog_frame()
    cat.to_csv(ROOT / "semantic_model" / "kpi_catalog.csv", index=False)
    return cat


def run(t, mmm_roi: float | None = None) -> pd.DataFrame:
    export_catalog()
    k = quarterly_kpis(t)
    k.to_csv(DATA_PROCESSED / "kpi_quarterly.csv", index=False)
    figures(t, k, mmm_roi)
    print(f"[kpis] {len(k)} quarters, catalog={len(KPI_CATALOG)} KPIs")
    return k
