-- =====================================================================
-- Semantic layer: governed KPI views (one definition per KPI).
-- Mirrors semantic_model/kpi_catalog.csv and semantic_model/measures.dax.
-- =====================================================================
USE SCHEMA CX_ANALYTICS;

-- Marketing efficiency by brand x quarter
CREATE OR REPLACE VIEW vw_kpi_marketing_quarterly AS
WITH spend AS (
    SELECT d.year_quarter, f.brand_key,
           SUM(f.spend_usd)                                                       AS spend_usd,
           SUM(IFF(c.promotion_type = 'Non-personal', f.spend_usd, 0))            AS np_spend_usd,
           SUM(f.clicks)                                                          AS clicks,
           SUM(f.impressions)                                                     AS impressions
    FROM fact_marketing_activity f
    JOIN dim_date d    ON d.date_key = f.date_key
    JOIN dim_channel c ON c.channel_key = f.channel_key
    GROUP BY 1, 2
), ord AS (
    SELECT d.year_quarter, o.brand_key,
           SUM(o.orders) AS orders, SUM(o.revenue_usd) AS revenue_usd, SUM(o.new_accounts) AS new_accounts
    FROM fact_orders o JOIN dim_date d ON d.date_key = o.date_key
    GROUP BY 1, 2
)
SELECT s.year_quarter, b.brand_name,
       s.spend_usd, o.orders, o.revenue_usd, o.new_accounts,
       s.spend_usd   / NULLIF(o.orders, 0)       AS cost_per_order,
       o.revenue_usd / NULLIF(s.spend_usd, 0)    AS revenue_per_mktg_dollar,
       s.clicks      / NULLIF(s.impressions, 0)  AS digital_engagement_rate,
       s.np_spend_usd / NULLIF(s.spend_usd, 0)   AS non_personal_share
FROM spend s
JOIN ord o       ON o.year_quarter = s.year_quarter AND o.brand_key = s.brand_key
JOIN dim_brand b ON b.brand_key = s.brand_key;

-- Customer experience by brand x quarter
CREATE OR REPLACE VIEW vw_kpi_cx_quarterly AS
SELECT d.year_quarter, b.brand_name,
       100 * (COUNT_IF(x.nps_0to10 >= 9) - COUNT_IF(x.nps_0to10 <= 6)) / NULLIF(COUNT(x.nps_0to10), 0) AS nps,
       AVG(IFF(x.csat_1to5 >= 4, 1, 0))                                         AS csat_top2_box,
       AVG(IFF(t.is_service, IFF(x.first_contact_resolved, 1, 0), NULL))        AS first_contact_resolution,
       AVG(IFF(t.is_service, x.resolution_hrs, NULL))                           AS avg_resolution_hrs,
       AVG(x.customer_effort_1to7)                                              AS customer_effort_score,
       COUNT(DISTINCT x.account_key)                                            AS active_accounts
FROM fact_cx_interaction x
JOIN dim_date d       ON d.date_key = x.date_key
JOIN dim_touchpoint t ON t.touchpoint_key = x.touchpoint_key
JOIN dim_brand b      ON b.brand_key = x.brand_key
GROUP BY 1, 2;

-- Incremental ROI is written back by the MMM job (see src/cx_suite/analytics/mmm.py)
CREATE TABLE IF NOT EXISTS mmm_channel_results (
    run_id          VARCHAR(40),
    run_ts          TIMESTAMP_NTZ,
    brand_code      VARCHAR(20),
    channel_code    VARCHAR(10),
    roi             NUMBER(8,3),
    roi_p05         NUMBER(8,3),
    roi_p95         NUMBER(8,3),
    marginal_roi    NUMBER(8,3),
    model_version   VARCHAR(20)
);
