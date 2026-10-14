"""Central configuration: paths, seeds, brand/channel catalogs, plot style.

All data in this repository is SYNTHETIC. Company and product names are used
only to make the scenarios realistic; no figures reflect actual performance.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_WAREHOUSE = ROOT / "data" / "warehouse"
FIG_ML = ROOT / "reports" / "figures" / "ml"
FIG_ANALYTICS = ROOT / "reports" / "figures" / "analytics"
METRICS = ROOT / "reports" / "metrics"
MODELS = ROOT / "models"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_WAREHOUSE, FIG_ML, FIG_ANALYTICS, METRICS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- customers
COMPANIES = {
    "Beckman Coulter": {
        "customer_types": ["Hospital Lab", "Reference Lab", "IDN Core Lab", "Physician Office Lab"],
        "product_lines": ["DxI Immunoassay", "DxH Hematology", "AU Chemistry", "Access Assays"],
    },
    "C2N Diagnostics": {
        "customer_types": ["Neurology Practice", "Memory Clinic", "Primary Care", "Geriatric Psychiatry"],
        "product_lines": ["PrecivityAD2 Blood Test"],
    },
}
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

# ---------------------------------------------------------------- marketing mix
BRANDS = {
    # brand_code: (name, company, clinical area, avg order value USD: reagent/consumable order for BEC, test for C2N)
    "BEC_IA": ("DxI Immunoassay", "Beckman Coulter", "Clinical Lab - Immunoassay", 650.0),
    "BEC_HEM": ("DxH Hematology", "Beckman Coulter", "Clinical Lab - Hematology", 480.0),
    "C2N_AD2": ("PrecivityAD2", "C2N Diagnostics", "Neurology - Alzheimer's Blood Test", 1450.0),
}
CHANNELS = {
    # code: (name, type, personal/non-personal)
    "REP": ("Field Rep Detailing", "Personal", "Field"),
    "EML": ("HCP / Lab Email", "Non-personal", "Owned"),
    "SRCH": ("Paid Search", "Non-personal", "Paid"),
    "DISP": ("Programmatic Display", "Non-personal", "Paid"),
    "SOC": ("LinkedIn / Paid Social", "Non-personal", "Paid"),
    "CONF": ("Congress & Trade Shows", "Personal", "Events"),
    "WEB": ("Peer Webinars", "Non-personal", "Owned"),
}

# ---------------------------------------------------------------- plotting
# Validated categorical palette (light mode) in fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a3a29c"
