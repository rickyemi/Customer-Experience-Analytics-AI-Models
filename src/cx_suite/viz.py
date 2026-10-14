"""Shared plotting style so every PNG reads as one system."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import GRID, PALETTE, SURFACE, TEXT, TEXT_2  # noqa: E402


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "figure.dpi": 110,
    })


def save(fig, path: Path, subtitle: str | None = None) -> Path:
    if subtitle:
        fig.text(0.01, 0.005, subtitle, fontsize=8, color=TEXT_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if subtitle else 0, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


set_style()
