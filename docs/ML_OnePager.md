# CX Predictive Algorithms: One-Pager
**Attrition Early-Warning + Next-Best-Action Uplift, with an A/B-tested MVP pilot**
Target customers: Beckman Coulter clinical labs (hospital, reference, IDN core, physician office) and C2N Diagnostics ordering practices (neurology, memory clinics, primary care). All data synthetic.

## 1. Objective / problem statement
Commercial teams spend outreach on accounts that would have ordered anyway and learn about at-risk accounts only after volume drops. Two products fix this:
* **Attrition early-warning**: predict 12-month account disengagement so Customer Success can act on the riskiest 10% (team capacity), with reasons a CSM can act on.
* **Next-Best-Action (NBA) uplift**: decide *whom* to contact so outreach changes behavior (persuadables), and suppress accounts where outreach backfires ("sleeping dogs").

## 2. Clinical and regulatory considerations
* Account-level, de-identified B2B data only; **no PHI** in features, logs or training sets (HIPAA). Not a clinical decision tool and not SaMD.
* Models choose **who and when**; content and claims come only from **MLR-approved** material and stay on-label (FDA-cleared IVD labeling for Beckman Coulter; validated performance claims for C2N's CLIA-lab test).
* Email only to consented contacts (CAN-SPAM, CCPA/CPRA); HCP transfers of value logged for **Sunshine Act / Open Payments**; outreach never tied to remuneration for ordering (Anti-Kickback Statute, AdvaMed Code).
* Responsible AI gates: subgroup performance parity (no group more than 0.05 AUC below overall), explainable reason codes, human-in-the-loop CSM review, model cards and versioned approvals.

## 3. Tools (end to end)
Snowflake / Redshift (feature store) · Python: pandas, scikit-learn, statsmodels, SciPy · MLflow (tracking + registry) · Airflow (orchestration) · Docker + GitHub Actions CI/CD · Salesforce / Veeva CRM and Marketo for activation · Power BI for monitoring dashboards · PSI drift monitor (Evidently-compatible).

## 4. Pipeline
`generate data -> engineer features -> train + calibrate -> evaluate (offline gates) -> offline policy evaluation -> A/B pilot -> deploy batch scores to CRM -> monitor drift -> retrain`

**Preprocessing**: unified *service-pain index* (instrument downtime for BEC, result turnaround for C2N, z-scored within business); NPS non-response flag + median imputation; log test volume; renewal-within-120-days flag; one-hot segments; leakage guard excludes labels, treatment and ground-truth columns (unit-tested).

## 5. Model development
* **Attrition**: logistic regression vs histogram gradient boosting, 5-fold stratified CV, Platt calibration, 30% hold-out. Ties go to the simpler model.
* **Uplift**: T-learner (two gradient-boosting models, treated vs control) trained on a historical 50/50 randomized campaign.
* **Pilot**: pre-registered A/B test (`configs/pilot_preregistration.yaml`). Arm A = today's rule (top 30% most engaged), Arm B = top 30% predicted uplift. Equal capacity isolates targeting quality.

## 6. Success metrics and results (hold-out / simulated pilot)
| Product | Metric | Target | Result |
|---|---|---|---|
| Attrition | ROC-AUC (CV) | >= 0.80 | **0.891** (0.889 ± 0.008) |
| Attrition | Top-decile lift / churners captured | >= 3x | **4.6x / 46%** |
| Attrition | Calibration (Brier) | < 0.10 | **0.083** |
| Attrition | Subgroup AUC range | within 0.05 | **0.877 to 0.913** |
| Uplift | Qini coefficient | > 0 | **0.55** |
| Uplift | True uplift, top 30% vs random | > 1.5x | **2.4x** (9.5 vs 3.9 pts) |
| Pilot | 60-day conversion lift (ITT) | +1.5 pts, p < 0.05 | **+2.75 pts** (95% CI 1.77 to 3.74), p = 2e-8 |
| Pilot | Unsubscribe guardrail | <= +0.3 pts | **+0.15 pts** (pass) |
| Monitoring | Score PSI | < 0.10 | 0.05 stable; email open rate PSI 0.42 triggers review |

## 7. Business impact
* At an assumed 20% CSM save rate, the top-decile worklist retains about **360 accounts per cycle (24,000-account book) versus about 80** from same-size untargeted outreach.
* Pilot: **+20.5% relative conversion** at equal outreach volume; about **$0.96M incremental margin per 10,000 accounts** (assumed $3,500 per converted account). Decision: **ship and scale to Wave 2**.
* The response model looked almost as good offline; uplift was chosen because it also suppresses negative-uplift accounts, protecting opt-out rates.

Figures: `reports/figures/ml/01-08_*.png` · Code: `src/cx_suite/ml/` · Notebook: `notebooks/02_cx_ml_models_and_pilot.ipynb`
