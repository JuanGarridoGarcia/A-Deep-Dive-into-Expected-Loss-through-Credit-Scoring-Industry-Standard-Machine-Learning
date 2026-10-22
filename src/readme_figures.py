"""
readme_figures.py

Charts used in the GitHub README (saved to assets/). They are drawn from the
outputs of the production notebooks, so the README figures always match the
latest run instead of being edited by hand.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS_DIR = "assets"

DARK = "#1F2937"
GREY = "#6B7280"
RED = "#B22222"
PD_DARK, PD_LIGHT = "#1A5FA5", "#B5D4F2"
LGD_DARK, LGD_LIGHT = "#6A3FA0", "#D0BCEB"
EL_DARK = "#2E7D32"

DISPLAY_NAMES = {
    "logistic_regression": "Logistic\nRegression",
    "ridge": "Ridge",
    "decision_tree": "Decision\nTree",
    "random_forest": "Random\nForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def display_name(model: str) -> str:
    """Readable model name (single line)."""
    return DISPLAY_NAMES.get(model, model).replace("\n", " ")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(colors=DARK)


def _save(fig, filename: str) -> str:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    path = os.path.join(ASSETS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    return path


def plot_pd_model_comparison(results: pd.DataFrame, selected: str,
                             filename: str = "pd_model_comparison.png"):
    """
    Test and cross-validated AUC-ROC per PD model, ordered by cross-validated AUC
    (the selection criterion). The selected model is highlighted.

    Points rather than bars: the models differ by thousandths of AUC, and bars on
    a truncated axis would exaggerate those differences.
    """
    df = results.sort_values("cv_auc", ascending=True)
    y = np.arange(len(df))
    colors = [PD_DARK if m == selected else PD_LIGHT for m in df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["cv_auc"], y, marker="D", s=60, facecolors="none", edgecolors=GREY,
               label="Cross-validated AUC (selection)", zorder=3)
    ax.scatter(df["auc"], y, s=90, color=colors, edgecolors=DARK, label="Test AUC", zorder=4)
    ax.set_yticks(y, [display_name(m) for m in df.index])

    lo, hi = df[["auc", "cv_auc"]].min().min(), df[["auc", "cv_auc"]].max().max()
    ax.set_xlim(lo - 0.01, hi + 0.025)
    for i, (auc, cv) in enumerate(zip(df["auc"], df["cv_auc"])):
        ax.text(max(auc, cv) + 0.002, i, f"{auc:.4f}  (CV {cv:.4f})", va="center",
                fontsize=10, fontweight="bold", color=DARK)

    ax.set_xlabel("AUC-ROC (models ranked by cross-validated AUC)", fontsize=12, color=DARK)
    ax.set_title("PD Model Comparison: AUC-ROC", fontsize=14, fontweight="bold", color=DARK, pad=12)
    ax.legend(loc="lower right")
    _style(ax)
    plt.tight_layout()
    return _save(fig, filename)


def plot_shap_comparison(importance_share: pd.DataFrame, filename: str = "shap_pd_comparison.png"):
    """
    Share of total mean |SHAP| per feature (%), one bar per model, features ordered
    by the first column (the selected model).
    """
    df = importance_share.sort_values(importance_share.columns[0], ascending=True)
    y = np.arange(len(df))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 1.5))
    for k, (col, color) in enumerate(zip(df.columns, [PD_DARK, PD_LIGHT])):
        ax.barh(y + (0.5 - k) * h, df[col], height=h, color=color, label=col)
    ax.set_yticks(y, df.index)
    ax.set_xlabel("Share of mean |SHAP| (%), one-hot columns summed per feature", color=DARK)
    ax.set_title("What drives the PD: SHAP importance in two model families",
                 fontsize=14, fontweight="bold", color=DARK, pad=12)
    ax.legend(loc="lower right")
    _style(ax)
    plt.tight_layout()
    return _save(fig, filename)


def plot_lgd_model_comparison(results: pd.DataFrame, selected: str, baseline_rmse: float = None,
                              filename: str = "lgd_model_comparison.png"):
    """
    Horizontal bars of test RMSE per LGD model, ordered by cross-validated RMSE
    (the selection criterion). The selected model is highlighted. If given,
    `baseline_rmse` (predicting the training mean for every loan) is drawn as a
    reference line, so the gain over a constant is visible.
    """
    df = results.sort_values("cv_rmse", ascending=False)
    colors = [LGD_DARK if m == selected else LGD_LIGHT for m in df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([DISPLAY_NAMES.get(m, m) for m in df.index], df["rmse"], color=colors, height=0.58)

    if baseline_rmse is not None:
        ax.axvline(baseline_rmse, color=RED, linestyle="--", lw=1.5,
                   label=f"Predict the mean: RMSE {baseline_rmse:.2f}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    right = max(df["rmse"].max(), baseline_rmse or 0)
    ax.set_xlim(0, right * 1.5)
    for i, (rmse, r2) in enumerate(zip(df["rmse"], df["r2"])):
        ax.text(right * 1.02, i, f"RMSE {rmse:.2f}  |  R² {r2:.4f}",
                va="center", fontsize=10, fontweight="bold", color=DARK)

    ax.set_xlabel("Test RMSE, percentage points (models ranked by cross-validated RMSE)",
                  fontsize=12, color=DARK)
    ax.set_title("LGD Model Comparison: RMSE and R²", fontsize=14, fontweight="bold",
                 color=DARK, pad=12)
    _style(ax)
    plt.tight_layout()
    return _save(fig, filename)


def plot_calibration_curves(prob_raw, prob_cal, y_true, model_label: str,
                            filename: str = "calibration_curve.png"):
    """Calibration curve on the test split before and after isotonic calibration."""
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, prob, title, label in [
        (axes[0], prob_raw, "Before calibration", f"{model_label} (raw)"),
        (axes[1], prob_cal, "After calibration", f"{model_label} (calibrated)"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", color=PD_DARK, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color=RED, label="Perfect calibration")
        ax.set_title(f"Calibration curve, test split: {title.lower()}", color=DARK)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend()
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return _save(fig, filename)


def plot_el_by_quartile(el_by_rate: pd.Series, portfolio_rate: float, rate_ranges: list,
                        filename: str = "el_by_quartile.png"):
    """
    Bar chart of the EL rate (%) per interest rate quartile.

    el_by_rate     : EL rate (%) indexed by Q1..Q4
    portfolio_rate : portfolio EL rate (%)
    rate_ranges    : interest rate range label per quartile, e.g. ["5-10%", ...]
    """
    labels = []
    for i, (q, rng) in enumerate(zip(el_by_rate.index, rate_ranges)):
        tag = "Lowest rates\n" if i == 0 else ("Highest rates\n" if i == len(rate_ranges) - 1 else "")
        labels.append(f"{q}\n{tag}({rng})")
    colors = [PD_LIGHT] * (len(el_by_rate) - 1) + [PD_DARK]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, el_by_rate.values, color=colors, width=0.55)
    ax.axhline(portfolio_rate, color=RED, linestyle="--", lw=1.5,
               label=f"Portfolio average ({portfolio_rate:.2f}%)")
    for i, v in enumerate(el_by_rate.values):
        ax.text(i, v + el_by_rate.max() * 0.015, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=DARK)

    ratio = el_by_rate.iloc[-1] / el_by_rate.iloc[0]
    ax.set_ylim(0, el_by_rate.max() * 1.25)
    ax.set_ylabel("Expected Loss Rate (%)", fontsize=12, color=DARK)
    ax.set_title("Expected Loss Rate by Interest Rate Quartile", fontsize=14,
                 fontweight="bold", color=DARK, pad=12)
    ax.legend(loc="upper left")
    fig.text(0.5, -0.02,
             f"Loans in the highest rate quartile carry {ratio:.1f}x the expected loss rate "
             f"of the lowest quartile", ha="center", fontsize=10, style="italic", color=GREY)
    _style(ax)
    plt.tight_layout()
    return _save(fig, filename)


def plot_key_metrics(pd_auc: float, pd_model: str, lgd_r2: float, lgd_model: str,
                     el_rate: float, el_total: float, n_loans: int, years: str,
                     filename: str = "key_metrics.png"):
    """Four headline tiles: PD AUC, LGD R², portfolio EL rate and portfolio size."""
    tiles = [
        ("PD MODEL", f"{pd_auc:.3f}", "AUC-ROC (test)", display_name(pd_model), PD_DARK, "#EEF3F9"),
        ("LGD MODEL", f"{lgd_r2:.3f}", "R² score (test)", display_name(lgd_model), LGD_DARK, "#F3F0F8"),
        ("EXPECTED LOSS", f"{el_rate:.2f}%", "portfolio rate", f"${el_total / 1e6:,.0f}M total",
         EL_DARK, "#EFF5EF"),
        ("PORTFOLIO", f"{n_loans / 1e6:.2f}M", "loans scored", years, DARK, "#EFEFF1"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (head, value, label, note, dark, light) in zip(axes, tiles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.86, color=light, lw=0))
        ax.add_patch(plt.Rectangle((0.02, 0.86), 0.96, 0.12, color=dark, lw=0))
        ax.text(0.5, 0.92, head, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")
        ax.text(0.5, 0.60, value, ha="center", va="center", color=dark,
                fontsize=38, fontweight="bold")
        ax.text(0.5, 0.34, label, ha="center", va="center", color=DARK, fontsize=14)
        ax.plot([0.25, 0.75], [0.25, 0.25], color=GREY, lw=0.6, alpha=0.6)
        ax.text(0.5, 0.15, note, ha="center", va="center", color=GREY,
                fontsize=12, style="italic")
    plt.tight_layout()
    return _save(fig, filename)


def interest_rate_ranges(df: pd.DataFrame, quartile_col: str = "interest_rate_q",
                         rate_col: str = "interest_rate") -> list:
    """Rounded min-max interest rate label per quartile, e.g. ['5-10%', ...]."""
    g = df.groupby(quartile_col, observed=True)[rate_col].agg(["min", "max"])
    return [f"{int(np.floor(lo))}-{int(np.ceil(hi))}%" for lo, hi in zip(g["min"], g["max"])]
