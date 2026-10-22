"""
lgd_analysis.py

Decomposition of the LGD target into the two things it mixes.

The LGD is measured against the amount funded (EAD at origination):

    LGD = (EAD - principal repaid - recoveries) / EAD = (1 - a) * (1 - r)

    a = principal repaid before default / EAD       (timing: how late the default came)
    r = recoveries / balance outstanding at default (recovery on what was still owed)

A borrower who defaults in month 3 and one who defaults in month 30 can share
the same application file and the same recovery rate, yet differ by tens of
points of LGD. These helpers measure how much of the LGD variance comes from
each part.
"""

import numpy as np
import pandas as pd

TARGET = "Target_Variable_Loss"


def lgd_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Components of the LGD for defaulted loans.

    Needs EAD, princ_rec, interest_rec, monthly_payment and the LGD target (in %).
    The recovery rate follows from the identity LGD = (1 - a)(1 - r), so it is
    exact by construction and needs no separate recovery column.

    Returns a frame with:
        lgd             : the target (%)
        a_repaid        : share of EAD repaid as principal before default
        r_recovered     : share of the balance at default that was recovered
        lgd_at_default  : 100 * (1 - r), the loss on the balance at default (%)
        months_paid     : approximate number of instalments paid before default
    """
    out = pd.DataFrame(index=df.index)
    out["lgd"] = df[TARGET]
    out["a_repaid"] = (df["princ_rec"] / df["EAD"]).clip(0, 1)
    balance_share = 1 - out["a_repaid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 1 - out["lgd"] / 100 / balance_share
    out["r_recovered"] = r.where(balance_share > 0).clip(0, 1)
    out["lgd_at_default"] = 100 * (1 - out["r_recovered"])
    out["months_paid"] = (df["princ_rec"] + df["interest_rec"]) / df["monthly_payment"]
    return out


def variance_explained(target: pd.Series, driver: pd.Series, bins: int = 50) -> float:
    """
    Share of the variance of `target` explained by `driver` alone: R² of the mean
    of `target` within quantile bins of `driver` (no functional form assumed).
    """
    ok = target.notna() & driver.notna()
    target, driver = target[ok], driver[ok]
    groups = pd.qcut(driver, bins, duplicates="drop")
    fitted = target.groupby(groups, observed=True).transform("mean")
    return float(1 - ((target - fitted) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def lgd_by_months_paid(components: pd.DataFrame,
                       edges=(0, 6, 12, 18, 24, 36, 48, 60, np.inf)) -> pd.DataFrame:
    """Mean LGD, share repaid and recovery rate by instalments paid before default."""
    groups = pd.cut(components["months_paid"], list(edges), include_lowest=True)
    return components.groupby(groups, observed=True).agg(
        loans=("lgd", "size"),
        lgd_mean=("lgd", "mean"),
        a_repaid_mean=("a_repaid", "mean"),
        r_recovered_mean=("r_recovered", "mean"),
    )
