-- MMM input at geo grain (brand x region x week), one spend column per channel.
-- Python equivalent: cx_suite.analytics.star_schema.model_frame()
USE SCHEMA CX_ANALYTICS;

CREATE OR REPLACE VIEW vw_mmm_model_frame AS
WITH s AS (
    SELECT f.brand_key, f.region_key, f.date_key, c.channel_code, SUM(f.spend_usd) AS spend_usd
    FROM fact_marketing_activity f JOIN dim_channel c ON c.channel_key = f.channel_key
    GROUP BY 1, 2, 3, 4
)
SELECT b.brand_code, p.region_key, p.date_key, d.week_start, d.is_holiday_week,
       p."'REP'" AS rep, p."'EML'" AS eml, p."'SRCH'" AS srch, p."'DISP'" AS disp,
       p."'SOC'" AS soc, p."'CONF'" AS conf, p."'WEB'" AS web,
       o.orders, o.revenue_usd
FROM s PIVOT (SUM(spend_usd) FOR channel_code IN ('REP','EML','SRCH','DISP','SOC','CONF','WEB')) p
JOIN fact_orders o ON o.brand_key = p.brand_key AND o.region_key = p.region_key AND o.date_key = p.date_key
JOIN dim_brand b   ON b.brand_key = p.brand_key
JOIN dim_date d    ON d.date_key = p.date_key;
