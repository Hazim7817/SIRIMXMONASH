"""Disproportionality statistics for spontaneous adverse-event reports.

Every report is counted once in a 2x2 table:

                    event    no event
    drug              a          b
    all other drugs   c          d

PRR (proportional reporting ratio) compares how often the event is reported
with the drug versus with everything else. ROR (reporting odds ratio) is the
odds-ratio version. Both flag drugs whose reports mention the event more
often than expected, which is a hypothesis about causation, not proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054

# Signal rules. Each takes the computed statistics and returns True/False.
CRITERIA = {
    # Evans et al. 2001, Pharmacoepidemiol Drug Saf 10:483-486.
    "evans": "PRR >= 2 and Yates chi-squared >= 4 and a >= 3",
    # Common in FAERS studies: lower bound of the ROR 95% CI above 1.
    "ror": "ROR lower 95% CI > 1 and a >= 3",
}


@dataclass(frozen=True)
class Table2x2:
    a: int
    b: int
    c: int
    d: int

    @classmethod
    def from_totals(cls, n_total: int, n_drug: int, n_event: int, n_drug_event: int) -> "Table2x2":
        """Build the table from four report counts.

        n_total: all reports; n_drug: reports mentioning the drug; n_event:
        reports with the event; n_drug_event: reports with both.
        """
        table = cls(
            a=n_drug_event,
            b=n_drug - n_drug_event,
            c=n_event - n_drug_event,
            d=n_total - n_drug - n_event + n_drug_event,
        )
        if min(table.a, table.b, table.c, table.d) < 0:
            raise ValueError(
                "Inconsistent counts (a cell came out negative): "
                f"total={n_total}, drug={n_drug}, event={n_event}, both={n_drug_event}"
            )
        return table

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d


@dataclass(frozen=True)
class Stats:
    prr: float | None
    prr_lower95: float | None
    prr_upper95: float | None
    ror: float | None
    ror_lower95: float | None
    ror_upper95: float | None
    chi2_yates: float | None
    haldane_corrected: bool


def _ci(ratio: float, se: float) -> tuple[float, float]:
    log_r = math.log(ratio)
    return math.exp(log_r - Z_95 * se), math.exp(log_r + Z_95 * se)


def compute(table: Table2x2) -> Stats:
    """PRR and ROR with 95% confidence intervals, and Yates chi-squared.

    If any cell is zero, 0.5 is added to every cell (Haldane-Anscombe
    correction) for the ratios and intervals, so they stay finite. With
    a = 0 there is no evidence of an association, and the ratios are
    reported as missing rather than as a corrected estimate.
    """
    a, b, c, d = table.a, table.b, table.c, table.d
    chi2 = chi2_yates(table)

    if a == 0 or (a + b) == 0 or (c + d) == 0:
        return Stats(None, None, None, None, None, None, chi2, False)

    haldane = 0 in (a, b, c, d)
    if haldane:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    prr = (a / (a + b)) / (c / (c + d))
    prr_se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    prr_lo, prr_hi = _ci(prr, prr_se)

    ror = (a * d) / (b * c)
    ror_se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ror_lo, ror_hi = _ci(ror, ror_se)

    return Stats(prr, prr_lo, prr_hi, ror, ror_lo, ror_hi, chi2, haldane)


def chi2_yates(table: Table2x2) -> float | None:
    """Pearson chi-squared with Yates continuity correction (1 df)."""
    a, b, c, d = table.a, table.b, table.c, table.d
    n = table.n
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return None
    diff = max(0.0, abs(a * d - b * c) - n / 2)
    return n * diff * diff / denom


def is_signal(table: Table2x2, stats: Stats, criterion: str) -> bool:
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r}; choose from {sorted(CRITERIA)}")
    if table.a < 3:
        return False
    if criterion == "evans":
        return (
            stats.prr is not None
            and stats.chi2_yates is not None
            and stats.prr >= 2
            and stats.chi2_yates >= 4
        )
    return stats.ror_lower95 is not None and stats.ror_lower95 > 1
