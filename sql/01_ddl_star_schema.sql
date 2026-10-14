-- =====================================================================
-- CX Analytics star schema  |  Snowflake dialect (Redshift: swap types)
-- Data layer: conformed dimensions + facts at explicit grain.
-- All data in this repository is synthetic.
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS CX_ANALYTICS;
USE SCHEMA CX_ANALYTICS;

-- ----------------------------- dimensions ----------------------------
CREATE OR REPLACE TABLE dim_date (
    date_key          NUMBER(8)    PRIMARY KEY,   -- yyyymmdd of week start (Monday)
    week_start        DATE         NOT NULL,
    year              NUMBER(4),
    quarter           NUMBER(1),
    month             NUMBER(2),
    iso_week          NUMBER(2),
    year_quarter      VARCHAR(7),
    is_holiday_week   BOOLEAN,
    congress          VARCHAR(30)                 -- ADLM / AAIC, AAN, CTAD
);

CREATE OR REPLACE TABLE dim_brand (
    brand_key            NUMBER       PRIMARY KEY,
    brand_code           VARCHAR(20)  UNIQUE NOT NULL,
    brand_name           VARCHAR(100),
    company              VARCHAR(100),
    clinical_area        VARCHAR(100),
    avg_order_value_usd  NUMBER(12,2)
);

CREATE OR REPLACE TABLE dim_channel (
    channel_key     NUMBER       PRIMARY KEY,
    channel_code    VARCHAR(10)  UNIQUE NOT NULL,
    channel_name    VARCHAR(100),
    promotion_type  VARCHAR(20),     -- Personal | Non-personal
    channel_group   VARCHAR(20)      -- Field | Owned | Paid | Events
);

CREATE OR REPLACE TABLE dim_region (
    region_key    NUMBER PRIMARY KEY,
    region_name   VARCHAR(50),
    territories   NUMBER,
    spend_weight  NUMBER(5,3)
);

CREATE OR REPLACE TABLE dim_campaign (
    campaign_key     NUMBER PRIMARY KEY,
    campaign_name    VARCHAR(100),
    brand_key        NUMBER REFERENCES dim_brand(brand_key),
    channel_key      NUMBER REFERENCES dim_channel(channel_key),
    fiscal_year      NUMBER(4),
    objective        VARCHAR(30),    -- Awareness | Consideration | Conversion | Retention
    target_audience  VARCHAR(100),
    mlr_approval_id  VARCHAR(30),    -- Medical-Legal-Regulatory review reference
    start_date       DATE,
    end_date         DATE
);

CREATE OR REPLACE TABLE dim_account (
    account_key        NUMBER PRIMARY KEY,
    account_id         VARCHAR(20) UNIQUE,  -- de-identified; no PHI is ever stored here
    primary_brand_key  NUMBER REFERENCES dim_brand(brand_key),
    customer_type      VARCHAR(50),
    region_key         NUMBER REFERENCES dim_region(region_key),
    account_tier       VARCHAR(1),
    value_decile       NUMBER(2),
    consent_email      BOOLEAN                -- CAN-SPAM / opt-in status drives eligibility
);

CREATE OR REPLACE TABLE dim_touchpoint (
    touchpoint_key   NUMBER PRIMARY KEY,
    touchpoint_name  VARCHAR(50),
    is_service       BOOLEAN,
    is_digital       BOOLEAN
);

-- ------------------------------- facts --------------------------------
-- Grain: week x brand x region x channel
CREATE OR REPLACE TABLE fact_marketing_activity (
    date_key      NUMBER(8) REFERENCES dim_date(date_key),
    brand_key     NUMBER    REFERENCES dim_brand(brand_key),
    region_key    NUMBER    REFERENCES dim_region(region_key),
    channel_key   NUMBER    REFERENCES dim_channel(channel_key),
    campaign_key  NUMBER    REFERENCES dim_campaign(campaign_key),
    spend_usd     NUMBER(14,2),
    impressions   NUMBER,
    clicks        NUMBER,
    engagements   NUMBER,       -- rep calls, booth leads, webinar attendees, email opens
    PRIMARY KEY (date_key, brand_key, region_key, channel_key)
)
CLUSTER BY (date_key, brand_key);

-- Grain: week x brand x region
CREATE OR REPLACE TABLE fact_orders (
    date_key      NUMBER(8) REFERENCES dim_date(date_key),
    brand_key     NUMBER    REFERENCES dim_brand(brand_key),
    region_key    NUMBER    REFERENCES dim_region(region_key),
    orders        NUMBER,
    revenue_usd   NUMBER(14,2),
    new_accounts  NUMBER,
    PRIMARY KEY (date_key, brand_key, region_key)
);

-- Grain: one customer interaction
CREATE OR REPLACE TABLE fact_cx_interaction (
    interaction_id          NUMBER PRIMARY KEY,
    date_key                NUMBER(8) REFERENCES dim_date(date_key),
    account_key             NUMBER    REFERENCES dim_account(account_key),
    brand_key               NUMBER    REFERENCES dim_brand(brand_key),
    touchpoint_key          NUMBER    REFERENCES dim_touchpoint(touchpoint_key),
    resolution_hrs          NUMBER(8,1),
    first_contact_resolved  BOOLEAN,
    csat_1to5               NUMBER(1),
    customer_effort_1to7    NUMBER(1),
    nps_0to10               NUMBER(2)
);

-- Load (example): COPY INTO fact_orders FROM @cx_stage/fact_orders.csv FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1);
