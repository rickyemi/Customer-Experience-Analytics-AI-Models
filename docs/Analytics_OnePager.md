# CX Performance Optimization: One-Pager
**Star-schema data model, governed KPI framework and in-house Marketing Mix Model (MMM)**
Brands: Beckman Coulter DxI Immunoassay and DxH Hematology (lab reagent/consumable orders), C2N Diagnostics PrecivityAD2 (blood tests ordered). 156 weeks x 5 regions x 7 channels. All data synthetic.

## 1. Objective / problem statement
Marketing investment (about $27M/yr across three brands) is planned on last year's split and vendor reports that disagree with each other. Leadership needs one governed answer to three questions: *What did each channel really return? Where should the next dollar go? Are customers' experiences improving?*

## 2. Clinical and regulatory considerations
* Aggregated, de-identified commercial data only; no patient or PHI data in any table (HIPAA). Account IDs are pseudonymous.
* Every campaign row carries an **MLR approval ID** so measured activity maps to approved, on-label content (FDA-cleared IVD labeling; validated LDT claims).
* Congress, webinar and rep activity feeds **Sunshine Act / Open Payments** reconciliation; budget shifts never alter HCP fair-market-value rules (AdvaMed Code).
* MMM outputs inform budget, not individual targeting; consent status in `dim_account` enforces email eligibility (CAN-SPAM, CCPA/CPRA).

## 3. Tools (end to end)
Snowflake / Redshift (data layer, `sql/01_ddl_star_schema.sql`) · dbt (proposed transforms + tests) · Python: pandas, SciPy, statsmodels · Power BI semantic model in DAX (`semantic_model/measures.dax`) · GitHub Actions (data-contract tests, CI) · Airflow weekly refresh · upgrade path: PyMC-Marketing / Google Meridian.

## 4. Data model and pipeline
* **Layer separation**: data (conformed star schema) -> semantic (12-KPI catalog, SQL views, DAX measures; one definition each) -> visualization (Power BI, PNGs).
* **Facts**: `fact_marketing_activity` (week x brand x region x channel), `fact_orders` (week x brand x region), `fact_cx_interaction` (one row per interaction). **Dimensions**: date, brand, channel, region, campaign, account, touchpoint.
* **Data contracts** (23 checks, run in CI): grain uniqueness, foreign-key resolution, non-negative measures.

**Preprocessing**: pivot spend to brand x region x week; geometric adstock (carry-over) and Hill saturation (diminishing returns) per channel; trend, two Fourier harmonics and holiday controls; 13-week time-ordered hold-out.

## 5. Model development
* **Geo-pooled MMM**: 780 region-weeks per brand with shared channel parameters; decay searched within business-plausible ranges; media effects constrained >= 0.
* **MAP estimation** with weakly informative ROI priors (mean 2.0, sd 2.5) and **geo lift-test priors** for low-spend channels (email, webinars) that weekly data alone cannot resolve: MMM + experiments triangulation.
* 200-draw moving-block bootstrap for ROI intervals; SLSQP budget optimizer at constant total spend (each channel 0.6x to 1.5x; field force at most -15% per cycle).

## 6. Success metrics and results
| Metric | Target | DxI | DxH | PrecivityAD2 |
|---|---|---|---|---|
| National R² (full) | >= 0.80 | 0.91 | 0.87 | 0.98 |
| 13-week hold-out MAPE | <= 10% | 1.8% | 2.1% | 4.2% |
| ROI vs known truth (correlation) | >= 0.80 | 0.88 | 0.87 | 0.94 |
| Reallocation gain, same budget | > 0 | +4.7% | +4.0% | +6.4% |

KPI scorecard FY25 vs FY24: NPS **41** (+18 pts), CSAT top-2-box **75%** (+10.7 pts), first-contact resolution **72%** (+8.2 pts), resolution time **18.7 h** (-17%), revenue +10.1%, cost per order +7.2% (flagged).

## 7. Business impact
* Portfolio incremental marketing ROI **1.88x**; reallocation worth about **$2.6M/yr incremental revenue at flat budget**, shifting toward paid search, HCP/lab email and peer webinars, and away from congress spend (all brands) and display or social where curves are saturated.
* One governed KPI layer replaces conflicting vendor reports; the scorecard ties CX program launch (mid-2024) to measurable service gains.
* Next step: run geo holdout tests on the two largest proposed shifts before full reallocation.

Figures: `reports/figures/analytics/01-09_*.png` · Code: `src/cx_suite/analytics/` · Notebook: `notebooks/03_analytics_star_schema_kpis_mmm.ipynb`
