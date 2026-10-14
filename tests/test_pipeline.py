"""Fast unit tests: run in CI on every pull request (pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cx_suite.data.generate_analytics_data import adstock, generate as gen_analytics, hill  # noqa: E402
from cx_suite.data.generate_ml_data import generate as gen_ml  # noqa: E402
from cx_suite.ml.features import EXCLUDE, NUMERIC, CATEGORICAL, feature_frame  # noqa: E402
from cx_suite.ml.monitoring import psi  # noqa: E402
from cx_suite.ml.uplift_model import qini_curve  # noqa: E402
from cx_suite.analytics.star_schema import validate  # noqa: E402


def test_ml_data_schema_and_rates():
    df = gen_ml(n=3000, seed=1)
    assert df.customer_id.is_unique
    assert 0.08 < df.churn_12m.mean() < 0.30
    assert set(df.nba_outreach.unique()) == {0, 1}


def test_no_leakage_into_features():
    X = feature_frame(gen_ml(n=500, seed=2))
    assert set(X.columns) == set(NUMERIC + CATEGORICAL)
    assert not set(X.columns) & set(EXCLUDE)


def test_adstock_and_hill():
    x = np.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(adstock(x, 0.5), [1, 0.5, 0.25, 0.125])
    assert abs(hill(np.array([2.0]), 2.0, 1.0)[0] - 0.5) < 1e-9


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1, 1, 5000)) > 0.25


def test_qini_ends_at_total_uplift():
    rng = np.random.default_rng(0)
    y, t, u = rng.integers(0, 2, 1000), rng.integers(0, 2, 1000), rng.random(1000)
    xs, qs = qini_curve(y, t, u)
    assert xs[0] == 0 and abs(xs[-1] - 1) < 1e-9


def test_star_schema_contracts():
    tables = gen_analytics()
    tables.pop("_truth")
    tables["dim_date"]["week_start"] = pd.to_datetime(tables["dim_date"]["week_start"])
    assert validate(tables).passed.all()
