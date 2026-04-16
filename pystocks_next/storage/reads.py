from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class PriceHistoryRead:
    """Consumer-oriented canonical price history contract."""

    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class DividendEventsRead:
    """Consumer-oriented canonical dividend history contract."""

    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SnapshotFeaturesRead:
    """Consumer-oriented canonical snapshot read contract."""

    frame: pd.DataFrame
