"""Build the three walkthrough notebooks (kept in code so they stay reviewable in PRs)."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, numpy as np
from IPython.display import Image, display
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)
FIG = ROOT / "reports" / "figures"
def show(p, w=950): display(Image(filename=str(FIG / p), width=w))"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    n["cells"] = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


nb1 = nb([
    ("md", "# 01 Synthetic data generation\n**CX Advanced Analytics & AI Suite** | all data are synthetic; company and product "
           "names are illustrative only.\n\nTwo datasets are produced:\n1. **AI/ML dataset**: one row per customer account "
           "(Beckman Coulter labs, C2N Diagnostics ordering practices) with churn and randomized-outreach labels.\n"
           "2. **Analytics warehouse**: a star schema (7 dimensions, 3 facts) for marketing-mix and CX KPIs."),
    ("code", SETUP),
    ("md", "## 1. AI/ML customer dataset"),
    ("code", "from cx_suite.data import generate_ml_data\nml = generate_ml_data.main()\nml.head()"),
    ("code", "ml.groupby('company').agg(accounts=('customer_id','count'), churn_rate=('churn_12m','mean'),\n"
             "    nps=('nps_score','mean'), email_open=('email_open_rate','mean')).round(3)"),
    ("md", "Missingness is intentional and meaningful: downtime applies to instrument customers (BEC), turnaround to "
           "send-out testing (C2N), and ~16% of accounts never answer NPS surveys."),
    ("code", "ml.isna().mean().loc[lambda s: s > 0].round(3)"),
    ("code", "# Randomized historical campaign: groups must be balanced on pre-treatment covariates\n"
             "ml.groupby('nba_outreach')[['tenure_months','email_open_rate','support_tickets_90d','digital_engagement_score']].mean().round(2)"),
    ("md", "## 2. Analytics warehouse (star schema)"),
    ("code", "from cx_suite.data import generate_analytics_data\nfrom cx_suite.analytics import star_schema\n"
             "generate_analytics_data.main()\nt = star_schema.load()\n{k: v.shape for k, v in t.items()}"),
    ("code", "star_schema.validate(t)"),
    ("code", "star_schema.draw_er_diagram()\nshow('analytics/01_star_schema_er_diagram.png')"),
    ("code", "t['fact_marketing_activity'].merge(t['dim_channel'], on='channel_key').groupby('channel_name').spend_usd.sum().div(3e6).round(2).rename('avg $M / yr')"),
])

nb2 = nb([
    ("md", "# 02 CX AI/ML models: MVP and pilot design\nThree products, each with explicit evaluation criteria:\n\n"
           "| Product | Question | Offline success criteria | Online (pilot) criteria |\n|---|---|---|---|\n"
           "| Attrition early-warning | Which accounts will disengage? | AUC >= 0.80, top-decile lift >= 3x, calibrated, subgroup AUC within 0.05 | Retention of contacted vs holdout accounts |\n"
           "| Next-Best-Action uplift | Whom should outreach target? | Qini > 0, monotone decile uplift | A/B: +1.5 pts 60-day conversion, guardrails pass |\n"
           "| Monitoring | Is the model still valid? | PSI < 0.10 on features and score | Retrain trigger at PSI > 0.25 |"),
    ("code", SETUP),
    ("code", "df = pd.read_csv(ROOT / 'data/raw/cx_customer_ml_dataset.csv')\nfrom cx_suite.ml.features import NUMERIC, CATEGORICAL, EXCLUDE\n"
             "print(len(NUMERIC), 'numeric +', len(CATEGORICAL), 'categorical features; excluded:', EXCLUDE)"),
    ("md", "## 1. Attrition early-warning model\nLogistic regression (interpretable baseline) vs gradient boosting, both "
           "calibrated. The champion is chosen on hold-out AUC; ties go to the simpler model."),
    ("code", "from cx_suite.ml import churn_model\nchurn = churn_model.run(df)\npd.DataFrame(churn['models']).T.round(3)"),
    ("code", "show('ml/01_churn_roc_pr.png'); show('ml/02_churn_calibration.png', 500)"),
    ("code", "show('ml/03_churn_gains_lift.png'); show('ml/04_churn_feature_importance.png', 700)"),
    ("code", "show('ml/05_churn_subgroup_auc.png', 700)"),
    ("md", "**Field-ready output**: a ranked worklist with plain-language reason codes, pushed to CRM for Customer Success."),
    ("code", "pd.read_csv(ROOT / 'data/processed/churn_worklist_top250.csv').head(10)"),
    ("md", "## 2. Next-Best-Action uplift model (T-learner)\nTrained on a historical 50/50 randomized outreach so the "
           "treatment effect is identifiable."),
    ("code", "from cx_suite.ml import uplift_model\nupl = uplift_model.run(df)\npd.DataFrame(upl['deciles']).round(3)"),
    ("code", "show('ml/06_nba_uplift_qini_deciles.png')"),
    ("md", "## 3. MVP pilot: A/B test design and readout\nPre-registration lives in `configs/pilot_preregistration.yaml`."),
    ("code", "from cx_suite.ml import pilot_design\npilot = pilot_design.run()\n"
             "{k: pilot[k] for k in ['power_analysis','abs_lift_pts','ci95_pts','relative_lift','p_value','guardrail_unsub_diff_pts','decision']}"),
    ("code", "show('ml/07_pilot_power_and_readout.png')"),
    ("md", "## 4. Post-deployment monitoring"),
    ("code", "from cx_suite.ml import monitoring\ndrift = monitoring.run()\nshow('ml/08_drift_monitor_psi.png', 700)"),
])

nb3 = nb([
    ("md", "# 03 Analytics: star schema, governed KPIs and in-house marketing mix\nLayering: **data** (star schema) "
           "-> **semantic** (KPI catalog, SQL views, DAX) -> **visualization** (Power BI / PNG)."),
    ("code", SETUP),
    ("code", "from cx_suite.analytics import star_schema, kpis, mmm\nt = star_schema.load()\nassert star_schema.validate(t).passed.all()\nkpis.catalog_frame()"),
    ("md", "## 1. MMM input frame (brand x region x week)"),
    ("code", "mf = star_schema.model_frame(t)\nmf.head()"),
    ("md", "## 2. Fit the geo-pooled MMM\nAdstock + Hill saturation, MAP estimation with weakly informative ROI priors, "
           "lift-test priors for low-spend channels, 13-week hold-out, block-bootstrap intervals."),
    ("code", "res = mmm.run(mf)\nS = res['summary']\npd.DataFrame({b: {'R2 (national)': S[b]['r2_full'], 'hold-out MAPE': S[b]['mape'],\n"
             "    'media share of orders': S[b]['media_share_of_orders'], 'reallocation uplift': S[b]['optimizer']['uplift_pct']}\n"
             "    for b in ['BEC_IA','BEC_HEM','C2N_AD2']}).T.round(3)"),
    ("code", "show('analytics/05_mmm_fit_holdout.png'); show('analytics/06_mmm_contribution_decomposition.png')"),
    ("code", "roi = pd.DataFrame({b: S[b]['roi'] for b in ['BEC_IA','BEC_HEM','C2N_AD2']}).round(2)\n"
             "truth = pd.DataFrame({b: S[b]['true_roi'] for b in ['BEC_IA','BEC_HEM','C2N_AD2']})\n"
             "pd.concat({'MMM ROI': roi, 'true ROI': truth}, axis=1)"),
    ("code", "show('analytics/07_mmm_roi_vs_truth.png'); show('analytics/08_mmm_response_curves_c2n.png', 800)"),
    ("md", "## 3. Budget optimization (constant total spend)"),
    ("code", "o = S['C2N_AD2']['optimizer']\npd.DataFrame({'current $/wk': o['current'], 'optimal $/wk': o['optimal']}).assign(change=lambda d: d['optimal $/wk']/d['current $/wk']-1).round(2)"),
    ("code", "show('analytics/09_budget_reallocation.png')"),
    ("md", "## 4. Governed KPI scorecard"),
    ("code", "k = kpis.run(t, S['portfolio']['incremental_roi'])\nk[['year_quarter','spend','orders','cpo','nps','csat_top2','fcr','resolution_hrs']].round(2)"),
    ("code", "show('analytics/02_kpi_scorecard.png'); show('analytics/03_cx_kpi_trends.png'); show('analytics/04_brand_spend_cpo.png', 800)"),
])

for name, n in [("01_synthetic_data_generation.ipynb", nb1), ("02_cx_ml_models_and_pilot.ipynb", nb2),
                ("03_analytics_star_schema_kpis_mmm.ipynb", nb3)]:
    nbf.write(n, NB / name)
    print("wrote", name)
