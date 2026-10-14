// Builds the three Word one-pagers in docs/ (node scripts/build_docx.js).
// Text supports **bold** inline markup.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  ShadingType, AlignmentType, LevelFormat, Footer, ImageRun,
} = require("docx");

const OUT = path.join(__dirname, "..", "docs");
const BLUE = "1C5CAB", INK = "0B0B0B", GRAY = "52514E", RULE = "B7D3F6", HEAD_FILL = "E8F0FB";
const FONT = "Calibri";
let BODY = 18; // half-points; set per document below
const PAGE_W = 12240, MARGIN = 720, CONTENT_W = PAGE_W - 2 * MARGIN;

function runs(text, opts = {}) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((p) => new TextRun({
    text: p.startsWith("**") ? p.slice(2, -2) : p, bold: p.startsWith("**") || opts.bold,
    font: FONT, size: opts.size || BODY, color: opts.color || INK, italics: opts.italics,
  }));
}
const title = (t) => new Paragraph({ spacing: { after: 20 }, children: runs(t, { size: 30, bold: true, color: BLUE }) });
const subtitle = (t) => new Paragraph({ spacing: { after: 60 }, children: runs(t, { size: 16, color: GRAY }) });
const H = (t) => new Paragraph({
  spacing: { before: 70, after: 25 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 1 } },
  children: runs(t.toUpperCase(), { size: 18, bold: true, color: BLUE }),
});
const P = (t) => new Paragraph({ spacing: { after: 30 }, children: runs(t) });
const B = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 10 }, children: runs(t) });

function table(rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 2, color: "D9D9D6" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        borders, width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 25, bottom: 25, left: 70, right: 70 },
        children: [new Paragraph({ children: runs(c, { size: BODY - 1, bold: i === 0 }) })],
      })),
    })),
  });
}
const FIG = path.join(__dirname, "..", "reports", "figures");
function figure(rel, widthIn, caption) {
  const buf = fs.readFileSync(path.join(FIG, rel));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);      // PNG IHDR dimensions
  const wpx = widthIn * 96, hpx = wpx * h / w;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 0 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: rel } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: runs(caption, { size: 14, color: GRAY, italics: true }) }),
  ];
}
const gap = () => new Paragraph({ spacing: { after: 20 }, children: [] });

function doc(children, footerText) {
  return new Document({
    styles: { default: { document: { run: { font: FONT, size: BODY } } } },
    numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 260, hanging: 180 } } } }] }] },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: 600, bottom: 560, left: MARGIN, right: MARGIN } } },
      footers: { default: new Footer({ children: [new Paragraph({ children: runs(footerText, { size: 14, color: GRAY }) })] }) },
      children,
    }],
  });
}

// ------------------------------------------------------------------ ML one-pager
BODY = 17;
const ml = doc([
  title("CX Predictive Algorithms: One-Pager"),
  subtitle("Attrition early-warning + Next-Best-Action uplift, proven through a pre-registered A/B pilot | Customers: Beckman Coulter clinical labs and C2N Diagnostics ordering practices | Synthetic data"),
  H("1. Objective / problem statement"),
  P("Outreach is spent on accounts that would order anyway, and at-risk accounts surface only after volume drops. **Attrition early-warning** ranks 12-month disengagement risk so Customer Success works the riskiest 10% with actionable reasons. **Next-Best-Action (NBA) uplift** picks whom to contact so outreach changes behavior, and suppresses accounts where it backfires."),
  H("2. Clinical and regulatory considerations"),
  B("Account-level, de-identified B2B data; **no PHI** in features, logs or training sets (HIPAA). Not a clinical decision tool and not SaMD."),
  B("Models choose **who and when**; content comes only from **MLR-approved**, on-label material (FDA-cleared IVD labeling; validated claims for C2N's CLIA-lab test)."),
  B("Consent-gated email (CAN-SPAM, CCPA/CPRA); **Sunshine Act / Open Payments** logging; no outreach tied to remuneration (Anti-Kickback Statute, AdvaMed Code)."),
  B("Responsible AI gates: subgroup AUC parity (within 0.05), reason codes, human-in-the-loop CSM review, model cards, versioned approvals."),
  H("3. Tools (end to end)"),
  P("Snowflake / Redshift feature tables | Python: pandas, scikit-learn, statsmodels, SciPy | MLflow tracking and registry | Airflow | Docker + GitHub Actions CI/CD | Salesforce / Veeva CRM and Marketo activation | Power BI monitoring | PSI drift monitor."),
  H("4. Pipeline and preprocessing"),
  P("**Generate > engineer > train + calibrate > offline gates > offline policy evaluation > A/B pilot > batch scores to CRM > drift monitor > retrain.** Unified service-pain index (instrument downtime for BEC, result turnaround for C2N, z-scored within business); NPS non-response flag with median imputation; log volume; renewal-within-120-days flag; one-hot segments; unit-tested leakage guard."),
  H("5. Model development"),
  B("**Attrition**: logistic regression vs gradient boosting, 5-fold stratified CV, Platt calibration, 30% hold-out; ties go to the simpler model (logistic regression won)."),
  B("**Uplift**: T-learner (treated and control gradient-boosting models) trained on a historical 50/50 randomized campaign."),
  B("**Pilot**: Arm A = today's rule (top 30% most engaged) vs Arm B = top 30% predicted uplift at equal capacity; one-sided z-test, alpha 0.05, power 0.80, MDE 1.5 pts (8,481 accounts per arm)."),
  H("6. Success metrics and results"),
  table([
    ["Product", "Metric", "Target", "Result"],
    ["Attrition", "ROC-AUC (5-fold CV)", ">= 0.80", "**0.891** (0.889 +/- 0.008)"],
    ["Attrition", "Top-decile lift / churners captured", ">= 3x", "**4.6x / 46%**"],
    ["Attrition", "Brier score; subgroup AUC range", "< 0.10; within 0.05", "**0.083; 0.877 to 0.913**"],
    ["Uplift", "Qini coefficient; true uplift top 30% vs random", "> 0; > 1.5x", "**0.55; 2.4x** (9.5 vs 3.9 pts)"],
    ["Pilot", "60-day conversion lift (ITT)", "+1.5 pts, p < 0.05", "**+2.75 pts** (95% CI 1.77 to 3.74), p = 2e-8"],
    ["Pilot", "Unsubscribe guardrail", "<= +0.3 pts", "**+0.15 pts** (pass)"],
    ["Monitoring", "Score PSI; feature PSI", "< 0.10; < 0.25", "0.05; email open rate 0.42 triggers review"],
  ], [1400, 3900, 1900, 3600]),
  ...figure("ml/03_churn_gains_lift.png", 6.0, "Attrition model: contacting the top 10% by risk captures 46% of churners; top decile churns at 4.6x the average (reports/figures/ml/03)."),
  H("7. Results / business impact"),
  B("At an assumed 20% save rate, the worklist retains about **360 accounts per cycle versus about 80** from same-size untargeted outreach (24,000-account book)."),
  B("Pilot: **+20.5% relative conversion** at equal outreach volume, about **$0.96M incremental margin per 10,000 accounts** ($3,500 per conversion assumed). Decision: **ship, scale to Wave 2**."),
  B("A response model looked nearly as good offline; uplift won because it also suppresses negative-uplift accounts and protects opt-out rates."),
], "Code: src/cx_suite/ml | Notebook: notebooks/02_cx_ml_models_and_pilot.ipynb | Figures: reports/figures/ml | Author: YO (github.com/rickyemi)");

// ------------------------------------------------------------------ Analytics one-pager
BODY = 17;
const an = doc([
  title("CX Performance Optimization: One-Pager"),
  subtitle("Star-schema data model, governed KPI framework and in-house Marketing Mix Model | Brands: DxI Immunoassay, DxH Hematology (Beckman Coulter), PrecivityAD2 (C2N Diagnostics) | 156 weeks x 5 regions x 7 channels | Synthetic data"),
  H("1. Objective / problem statement"),
  P("About **$27M/yr** of marketing is planned on last year's split and conflicting vendor reports. Leadership needs one governed answer: what did each channel return, where should the next dollar go, and is customer experience improving?"),
  H("2. Clinical and regulatory considerations"),
  B("Aggregated, de-identified commercial data; no patient data or PHI in any table; pseudonymous account IDs (HIPAA)."),
  B("Every campaign row carries an **MLR approval ID**, tying measured activity to approved, on-label content (FDA-cleared IVD labeling; validated LDT claims)."),
  B("Congress, webinar and rep activity reconciles to **Sunshine Act / Open Payments**; budget shifts never change fair-market-value rules (AdvaMed Code). Consent flag enforces email eligibility."),
  H("3. Tools (end to end)"),
  P("Snowflake / Redshift data layer | dbt transforms and tests (proposed) | Python: pandas, SciPy, statsmodels | Power BI semantic model (DAX) | GitHub Actions data-contract CI | Airflow weekly refresh | upgrade path: PyMC-Marketing or Google Meridian."),
  H("4. Data model and preprocessing"),
  B("**Layer separation**: data (conformed star schema) > semantic (12-KPI catalog, SQL views, DAX; one definition per KPI) > visualization (Power BI)."),
  B("**Facts**: marketing activity (week x brand x region x channel), orders (week x brand x region), CX interactions (one row each). **Dims**: date, brand, channel, region, campaign, account, touchpoint. **23 data-contract checks** (grain, foreign keys, non-negative measures)."),
  B("Spend pivoted to brand x region x week; geometric **adstock** (carry-over) and **Hill saturation** (diminishing returns); trend, Fourier seasonality and holiday controls; 13-week time-ordered hold-out."),
  H("5. Model development"),
  B("**Geo-pooled MMM**: 780 region-weeks per brand, shared channel parameters, decay searched within business-plausible ranges, media effects constrained >= 0."),
  B("**MAP estimation** with weakly informative ROI priors and **geo lift-test priors** for low-spend channels (email, webinars): MMM and experiments triangulated."),
  B("200-draw block bootstrap for ROI intervals; SLSQP optimizer at flat total budget (channels 0.6x to 1.5x; field force at most -15% per cycle)."),
  H("6. Success metrics and results"),
  table([
    ["Metric", "Target", "DxI", "DxH", "PrecivityAD2"],
    ["National R-squared", ">= 0.80", "0.91", "0.87", "0.98"],
    ["13-week hold-out MAPE", "<= 10%", "1.8%", "2.1%", "4.2%"],
    ["ROI vs known truth (correlation)", ">= 0.80", "0.88", "0.87", "0.94"],
    ["Incremental revenue from reallocation, same budget", "> 0", "+4.7%", "+4.0%", "+6.4%"],
  ], [4300, 1300, 1700, 1700, 1800]),
  ...figure("analytics/07_mmm_roi_vs_truth.png", 7.0, "Channel ROI: MMM estimate with 90% interval vs known synthetic truth (reports/figures/analytics/07)."),
  P("**KPI scorecard FY25 vs FY24**: NPS 41 (+18 pts) | CSAT top-2-box 75% (+10.7 pts) | first-contact resolution 72% (+8.2 pts) | resolution time 18.7 h (-17%) | revenue +10.1% | cost per order +7.2% (flagged for action)."),
  H("7. Results / business impact"),
  B("Portfolio incremental marketing ROI **1.88x**; reallocation worth about **$2.6M/yr incremental revenue at flat budget**, shifting toward paid search, HCP/lab email and peer webinars and away from congress spend and saturated display or social."),
  B("One governed KPI layer replaces conflicting vendor reports; the scorecard links the mid-2024 CX program to measurable service gains."),
  B("Next: geo holdout tests on the two largest proposed shifts before full reallocation."),
], "Code: src/cx_suite/analytics | SQL: sql/ | DAX: semantic_model/ | Notebook: notebooks/03 | Figures: reports/figures/analytics | Author: YO (github.com/rickyemi)");

// ------------------------------------------------------------------ Strategy brief
BODY = 19;
const st = doc([
  title("In-House Marketing Mix Strategy for Diagnostics CX"),
  subtitle("Director, CX Advanced Analytics & AI point of view | Customers: Beckman Coulter clinical labs and C2N Diagnostics neurology and primary-care practices | Illustrative, synthetic figures"),
  H("Strategic intent"),
  P("Own the measurement engine instead of renting it. An in-house marketing mix solution gives brand teams a **weekly, explainable, governed** view of what each channel returns, built in our own cloud stack and wired to the same KPI layer leadership already trusts. Vendors still supply data and specialist tests; **decisions, assumptions and model code stay with us.**"),
  H("Two customer journeys, one framework"),
  table([
    ["", "Clinical lab (Beckman Coulter)", "Neurology / PCP practice (C2N Diagnostics)"],
    ["Buyer and cycle", "Lab director, procurement, value committee; 12 to 36 month cycles", "Neurologist, memory clinic, PCP; new category, fast trial"],
    ["What moves them", "Uptime, turnaround, service experience, reagent economics", "Clinical education, ordering ease, results clarity, coverage"],
    ["Channel roles", "Field + congress (ADLM) open doors; email, webinars, search sustain pull-through", "Peer webinars and search build demand; reps convert; email drives repeat ordering"],
    ["North-star outcome", "Retained and expanded reagent volume per account", "New ordering practices and repeat ordering rate"],
  ], [1900, 4450, 4450]),
  H("Measurement architecture: triangulate, never single-source"),
  B("**MMM (quarterly budget decisions)**: geo-pooled, adstock and saturation, refreshed weekly; answers how much per channel."),
  B("**Experiments (truth anchor)**: geo holdouts and A/B tests calibrate the MMM; every major reallocation earns a test first."),
  B("**Journey analytics (in-flight optimization)**: campaign and account-level signals tune creative, cadence and targeting between MMM cycles."),
  H("KPI framework"),
  table([
    ["Tier", "KPIs (governed definitions, one source of truth)", "FY target"],
    ["North star", "Incremental marketing ROI (MMM); Net Promoter Score", ">= 2.0x; >= 40"],
    ["Drivers", "Cost per order; new accounts; repeat-order rate; digital engagement rate; non-personal promotion share", "-5% CPO; +10% accounts; 35 to 45% NPP"],
    ["Diagnostics", "CSAT top-2-box; first-contact resolution; resolution hours; customer effort; active accounts", ">= 75%; >= 70%; <= 18 h; <= 3.0"],
  ], [1500, 6900, 2400]),
  H("How we judge the capability itself"),
  B("**Adoption**: at least 80% of brand budget decisions cite MMM scenarios by year end."),
  B("**Accuracy and calibration**: hold-out MAPE under 10%; MMM ROI inside lift-test confidence intervals for tested channels."),
  B("**Speed**: weekly refresh, any budget scenario returned in under one day. **Value**: realized incremental revenue vs plan, audited quarterly with Finance."),
  H("Roadmap"),
  B("**0 to 90 days, Foundation**: star schema and data contracts, KPI catalog sign-off with Brand, Finance, Compliance; baseline MMM; Power BI scorecard live."),
  B("**3 to 6 months, Prove**: two geo holdout tests (email, congress); first 10% budget reallocation; NBA uplift pilot with pre-registered A/B design."),
  B("**6 to 12 months, Scale**: Bayesian MMM with experiment priors, always-on optimizer in planning cycle, MMM outputs feeding NBA channel mix."),
  H("Governance, build vs buy, risks"),
  P("**Build** the model, KPI layer and optimizer; **buy** media data connectors and specialist testing. MLR approval IDs on every campaign, no PHI, Sunshine Act reconciliation and AdvaMed-aligned interactions are design requirements. **Key risks**: weak signal in small channels (mitigated by lift-test priors), stakeholder trust (mitigated by transparent assumptions and holdout tests), data latency (mitigated by refresh SLAs and lineage)."),
], "Supporting code and results: CX Advanced Analytics & AI Technical Suite repository (github.com/rickyemi) | Prepared by YO");

(async () => {
  for (const [name, d] of [["ML_OnePager.docx", ml], ["Analytics_OnePager.docx", an], ["CX_Marketing_Mix_Strategy_OnePager.docx", st]]) {
    fs.writeFileSync(path.join(OUT, name), await Packer.toBuffer(d));
    console.log("wrote", name);
  }
})();
