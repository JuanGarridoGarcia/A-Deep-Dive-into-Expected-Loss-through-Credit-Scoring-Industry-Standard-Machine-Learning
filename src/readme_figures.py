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
        ("LGD MODEL", f"{lgd_r2:.3f}", "R² score (test)", f"{display_name(lgd_model)} (mean: 0)",
         LGD_DARK, "#F3F0F8"),
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
    """Min-max interest rate label per quartile with one decimal, e.g. ['5.3-9.9%', ...].

    Whole-number rounding made neighbouring quartiles look overlapping (9-13%, 12-16%).
    """
    g = df.groupby(quartile_col, observed=True)[rate_col].agg(["min", "max"])
    return [f"{lo:.1f}-{hi:.1f}%" for lo, hi in zip(g["min"], g["max"])]


def plot_pipeline(v: dict, filename: str = "pipeline.png"):
    """
    Overview diagram of the pipeline for the README, with the numbers of the run.

    `v` holds the values shown in the boxes: n_loans, years, snapshot, n_matured,
    n_defaults, n_features_pd, n_features_lgd, pd_model, pd_auc, pd_mean_test,
    default_rate_test, lgd_model, lgd_r2, lgd_rmse, lgd_baseline_rmse,
    lgd_pred_test, lgd_obs_test, el_rate, el_total, el_ratio_test,
    el_rate_matured, el_rate_running.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 23))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 23)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge, face, title_color=None, size=12.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                    facecolor=face, edgecolor=edge, lw=2))
        ax.text(x + w / 2, y + h - 0.38, title, ha="center", va="center", fontsize=size + 1.5,
                fontweight="bold", color=title_color or edge)
        ax.text(x + w / 2, y + (h - 0.62) / 2, "\n".join(lines), ha="center", va="center",
                fontsize=size - 1.5, color=DARK, linespacing=1.6)

    def arrow(a, b, color=DARK):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=20, color=color, lw=2.2))

    ax.text(6.5, 22.45, "Credit Risk Modeling Pipeline", ha="center", fontsize=22, fontweight="bold", color=DARK)
    ax.text(6.5, 21.95, "Lifetime Expected Loss with the Basel decomposition EL = PD \u00d7 LGD \u00d7 EAD",
            ha="center", fontsize=12.5, style="italic", color=GREY)

    box(3.0, 20.0, 7.0, 1.4, "RAW DATA",
        [f"{v['n_loans'] / 1e6:.2f}M Lending Club loans  \u00b7  issued {v['years']}  \u00b7  snapshot {v['snapshot']}"],
        GREY, "#F3F4F6", DARK)
    arrow((6.5, 19.95), (6.5, 19.35))
    box(3.0, 17.75, 7.0, 1.55, "DATA DIAGNOSIS",
        ["LGD target: net loss / EAD  \u00b7  MNAR imputation",
         f"development sample: {v['n_matured'] / 1e3:,.0f}k matured loans (outcome complete)"],
        GREY, "#F3F4F6", DARK)
    arrow((6.5, 17.7), (6.5, 17.1))
    box(3.0, 15.5, 7.0, 1.55, "FEATURE SELECTION",
        ["Mutual information  \u00b7  permutation  \u00b7  Lasso  \u00b7  Boruta  \u00b7  SHAP cutoff",
         "leakage, vintage-dependent and survival-bias exclusions"],
        GREY, "#F3F4F6", DARK)

    arrow((6.5, 15.45), (3.3, 14.15))
    arrow((6.5, 15.45), (9.7, 14.15))
    ax.text(2.0, 14.75, "PD  \u00b7  all matured loans", ha="center", fontsize=12.5, fontweight="bold", color=PD_DARK)
    ax.text(11.0, 14.75, f"LGD  \u00b7  {v['n_defaults'] / 1e3:,.0f}k defaulted, matured",
            ha="center", fontsize=12.5, fontweight="bold", color=LGD_DARK)

    box(0.6, 12.6, 5.4, 1.5, "EDA & TREATMENT",
        [f"{v['n_features_pd']} features  \u00b7  caps and imputation", "fitted on matured loans only"],
        PD_DARK, "#DCE9F6")
    box(7.0, 12.6, 5.4, 1.5, "EDA & TREATMENT",
        [f"{v['n_features_lgd']} features  \u00b7  caps and imputation", "fitted on the training split only"],
        LGD_DARK, "#ECE4F6")
    arrow((3.3, 12.55), (3.3, 12.0), PD_DARK)
    arrow((9.7, 12.55), (9.7, 12.0), LGD_DARK)

    box(0.6, 10.1, 5.4, 1.85, "PD MODEL",
        ["6 classifiers  \u00b7  selected by CV AUC", f"selected: {display_name(v['pd_model'])}",
         f"test AUC {v['pd_auc']:.3f}"], PD_DARK, "#C9DDF2")
    box(7.0, 10.1, 5.4, 1.85, "LGD MODEL",
        ["6 regressors  \u00b7  selected by CV RMSE", f"selected: {display_name(v['lgd_model'])}",
         f"test R\u00b2 {v['lgd_r2']:.3f}  \u00b7  RMSE {v['lgd_rmse']:.1f} (mean: {v['lgd_baseline_rmse']:.1f})"],
        LGD_DARK, "#DDD0EE")
    arrow((3.3, 10.05), (3.3, 9.5), PD_DARK)
    arrow((9.7, 10.05), (9.7, 9.5), LGD_DARK)

    box(0.6, 7.85, 5.4, 1.6, "CALIBRATION",
        ["isotonic, on a separate split", f"test: mean PD {v['pd_mean_test']:.3f} vs default rate {v['default_rate_test']:.3f}"],
        PD_DARK, "#DCE9F6")
    box(7.0, 7.85, 5.4, 1.6, "CALIBRATION CHECK",
        ["mean level by segment", f"test: predicted {v['lgd_pred_test']:.1f} vs observed {v['lgd_obs_test']:.1f}"],
        LGD_DARK, "#ECE4F6")
    arrow((3.3, 7.8), (5.3, 6.75), PD_DARK)
    arrow((9.7, 7.8), (7.7, 6.75), LGD_DARK)

    ax.add_patch(FancyBboxPatch((3.0, 5.35), 7.0, 1.35, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=EL_DARK, edgecolor=EL_DARK))
    ax.text(6.5, 6.25, "EXPECTED LOSS", ha="center", va="center", fontsize=15, fontweight="bold", color="white")
    ax.text(6.5, 5.75, f"EL = PD \u00d7 LGD \u00d7 EAD  \u00b7  {v['n_loans'] / 1e6:.2f}M loans scored",
            ha="center", va="center", fontsize=12, color="white")
    arrow((6.5, 5.3), (6.5, 4.75), EL_DARK)

    ax.add_patch(FancyBboxPatch((1.6, 1.4), 9.8, 3.3, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="white", edgecolor=EL_DARK, lw=2))
    ax.text(6.5, 4.2, "PORTFOLIO RESULTS", ha="center", fontsize=15, fontweight="bold", color=EL_DARK)
    stats = [(f"{v['el_rate']:.2f}%", "EL rate"), (f"${v['el_total'] / 1e6:,.0f}M", "total EL"),
             (f"{v['el_ratio_test']:.2f}", "EL / realised loss (test)"), (f"{v['n_loans'] / 1e6:.2f}M", "loans scored")]
    for i, (val, lab) in enumerate(stats):
        x = 2.85 + i * 2.45
        ax.text(x, 3.15, val, ha="center", fontsize=19, fontweight="bold", color=DARK)
        ax.text(x, 2.55, lab, ha="center", fontsize=10.5, color=GREY)
    ax.text(6.5, 1.85, f"lifetime EL rate: matured loans {v['el_rate_matured']:.2f}%  \u00b7  "
            f"loans still running {v['el_rate_running']:.2f}%", ha="center", fontsize=10.5,
            style="italic", color=GREY)
    return _save(fig, filename)
