from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass(frozen=True, slots=True)
class PortfolioInputBundle:
    """Explicit portfolio inputs for optimizer-facing workflows."""

    expected_returns: pd.DataFrame = field(default_factory=_empty_frame)
    covariance: pd.DataFrame = field(default_factory=_empty_frame)
    exposures: pd.DataFrame = field(default_factory=_empty_frame)
    eligibility: pd.DataFrame = field(default_factory=_empty_frame)
