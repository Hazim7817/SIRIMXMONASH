import math

import pytest

from loaders.disproportionality import Table2x2, compute, is_signal

# Expected values were cross-checked against scipy.stats
# (odds_ratio, chi2_contingency with correction=True).


def test_from_totals_builds_2x2():
    t = Table2x2.from_totals(n_total=100_000, n_drug=1_000, n_event=1_020, n_drug_event=20)
    assert (t.a, t.b, t.c, t.d) == (20, 980, 1_000, 98_000)


def test_from_totals_rejects_inconsistent_counts():
    with pytest.raises(ValueError):
        Table2x2.from_totals(n_total=100, n_drug=10, n_event=5, n_drug_event=6)


@pytest.mark.parametrize(
    "cells, ror, ror_ci, chi2, prr",
    [
        ((20, 980, 1_000, 98_000), 2.0, (1.27900, 3.12745), 8.65333, 1.98),
        ((3, 50, 400, 600_000), 90.0, (27.95661, 289.73467), 170.8679, 84.96226),
        ((150, 10_000, 20_000, 15_000_000), 11.25, (9.56917, 13.22607), 1359.84471, 11.09852),
    ],
)
def test_statistics_match_reference_values(cells, ror, ror_ci, chi2, prr):
    s = compute(Table2x2(*cells))
    assert s.ror == pytest.approx(ror, rel=1e-6)
    assert (s.ror_lower95, s.ror_upper95) == pytest.approx(ror_ci, rel=1e-5)
    assert s.chi2_yates == pytest.approx(chi2, rel=1e-6)
    assert s.prr == pytest.approx(prr, rel=1e-5)
    assert s.prr_lower95 < s.prr < s.prr_upper95
    assert not s.haldane_corrected


def test_prr_interval_formula():
    a, b, c, d = 20, 980, 1_000, 98_000
    s = compute(Table2x2(a, b, c, d))
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    assert s.prr_lower95 == pytest.approx(s.prr * math.exp(-1.959964 * se), rel=1e-6)


def test_zero_cell_uses_haldane_correction():
    s = compute(Table2x2(5, 0, 30, 1_000))
    assert s.haldane_corrected
    assert s.ror == pytest.approx((5.5 * 1000.5) / (0.5 * 30.5))


def test_no_drug_event_reports_gives_no_ratio_and_no_signal():
    t = Table2x2(0, 500, 1_000, 98_000)
    s = compute(t)
    assert s.prr is None and s.ror is None
    assert not is_signal(t, s, "evans") and not is_signal(t, s, "ror")


def test_signal_rules():
    strong = Table2x2(150, 10_000, 20_000, 15_000_000)
    assert is_signal(strong, compute(strong), "evans")
    assert is_signal(strong, compute(strong), "ror")

    weak = Table2x2(20, 980, 1_000, 98_000)  # PRR 1.98: just under Evans' threshold
    assert not is_signal(weak, compute(weak), "evans")
    assert is_signal(weak, compute(weak), "ror")  # ROR CI 1.28-3.13 excludes 1

    too_few = Table2x2(2, 10, 1_000, 1_000_000)  # huge ratio but only 2 reports
    assert not is_signal(too_few, compute(too_few), "ror")


def test_unknown_criterion():
    t = Table2x2(5, 5, 5, 5)
    with pytest.raises(ValueError):
        is_signal(t, compute(t), "magic")
