"""
Publication figure style for the anomaly-detection paper.

Usage:
    import figs_style as FS
    FS.set_style()
    ...
    FS.save(fig, "path/to/figure_name")   # emits .pdf + .png
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── colour tokens ──────────────────────────────────────────────────
INK = "#1A1A1A"
MUTED = "#6B7280"
GRID = "#E5E7EB"

PRIMARY = "#2A6F97"      # teal-blue  — main series / normal class
PRIMARY_L = "#A9CCE3"    # light teal — fills, shading
ACCENT = "#E1812C"       # amber      — focal element (e.g. "Ours", operating pt)
ACCENT_L = "#F4C89B"     # light amber — shading behind accent

GOOD = "#2E7D5B"         # green  — positive outcome
BAD = "#B4483C"          # red    — negative outcome / anomaly class

BG = "#FFFFFF"            # background
NEUTRAL = "#D1D5DB"       # light grey — TN / background class

NEUTRALS = ["#4C6E8A", "#7FA3B8", "#B7C9D6", "#9AA5B1", "#C3CAD3"]

SEQ_BLUE = ["#F2F7FB", "#D8E7F1", "#A9CCE3", "#6FA8C9", "#2A6F97"]

# For confusion matrix — a 5-step sequential blue
from matplotlib.colors import LinearSegmentedColormap
SEQ_BLUE_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE, N=256)


def set_style():
    """Apply publication rcParams globally.  Call once at script start."""
    mpl.rcParams.update({
        # fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        # colours
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        # spines & grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # resolution
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def save(fig, path_noext):
    """Save *fig* as both PDF (for LaTeX) and PNG (for preview)."""
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png")
    plt.close(fig)


# ── corrected d-prime ──────────────────────────────────────────────
def d_prime(pos, neg, eps=1e-6):
    """
    Pooled-variance d-prime (signal-detection theory).

    Uses the average of the two class variances as denominator, with a
    relative floor so the value stays finite when one class has near-zero
    spread.  Capped at 10.0 for display.
    """
    mu_diff = abs(np.mean(pos) - np.mean(neg))
    pooled_sd = np.sqrt(0.5 * (np.var(pos) + np.var(neg)))
    pooled_sd = max(pooled_sd, eps * (mu_diff + eps))
    raw = mu_diff / pooled_sd
    return min(raw, 10.0)


def d_prime_label(val):
    """Format d-prime for annotation: show '> 10' if capped."""
    if val >= 10.0:
        return "d′ > 10"
    return f"d′ = {val:.2f}"
