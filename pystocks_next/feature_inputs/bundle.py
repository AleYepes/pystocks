from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass(frozen=True, slots=True)
class AnalysisInputBundle:
    """Stable analysis-facing inputs owned by the feature-input stage."""

    prices: pd.DataFrame = field(default_factory=_empty_frame)
    price_eligibility: pd.DataFrame = field(default_factory=_empty_frame)
    dividends: pd.DataFrame = field(default_factory=_empty_frame)
    dividend_summary: pd.DataFrame = field(default_factory=_empty_frame)
    snapshot_features: pd.DataFrame = field(default_factory=_empty_frame)
    snapshot_diagnostics: pd.DataFrame = field(default_factory=_empty_frame)
    risk_free_daily: pd.DataFrame = field(default_factory=_empty_frame)
    macro_features: pd.DataFrame = field(default_factory=_empty_frame)
