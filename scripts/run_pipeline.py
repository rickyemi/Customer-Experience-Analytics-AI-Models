"""End-to-end pipeline: data -> ML (churn, uplift, pilot, drift) -> analytics (schema, MMM, KPIs).

Usage:
    python scripts/run_pipeline.py            # everything
    python scripts/run_pipeline.py --only ml
    python scripts/run_pipeline.py --only analytics
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cx_suite.analytics import kpis, mmm, star_schema  # noqa: E402
from cx_suite.data import generate_analytics_data, generate_ml_data  # noqa: E402
from cx_suite.ml import churn_model, monitoring, pilot_design, uplift_model  # noqa: E402


def run_ml():
    df = generate_ml_data.main()
    churn_model.run(df)
    uplift_model.run(df)
    pilot_design.run()
    monitoring.run()


def run_analytics():
    generate_analytics_data.main()
    t = star_schema.load()
    checks = star_schema.validate(t)
    if not checks.passed.all():
        raise SystemExit(f"Data contract failures:\n{checks[~checks.passed]}")
    print(f"[schema] {len(checks)} data-contract checks passed")
    star_schema.draw_er_diagram()
    res = mmm.run(star_schema.model_frame(t))
    kpis.run(t, res["summary"]["portfolio"]["incremental_roi"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ml", "analytics"])
    a = ap.parse_args()
    t0 = time.time()
    if a.only in (None, "ml"):
        run_ml()
    if a.only in (None, "analytics"):
        run_analytics()
    print(f"Done in {time.time() - t0:.0f}s. Figures in reports/figures, metrics in reports/metrics.")
