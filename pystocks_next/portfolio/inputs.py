from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EXPECTED_RETURN_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "conid",
    "expected_return",
)

COVARIANCE_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "left_conid",
    "right_conid",
    "covariance",
)

EXPOSURE_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "conid",
    "factor_name",
    "exposure",
)

ELIGIBILITY_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "conid",
    "is_eligible",
    "eligibility_reason",
)


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _normalize_frame(
    frame: pd.DataFrame | None,
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    source = _empty_frame(columns) if frame is None else frame.copy()
    missing_columns = [column for column in columns if column not in source.columns]
    if missing_columns and not source.empty:
        missing = ", ".join(missing_columns)
        raise ValueError(f"frame is missing required columns: {missing}")
    return source.reindex(columns=pd.Index(columns)).copy()


@dataclass(frozen=True, slots=True)
class PortfolioInputBundle:
    """Explicit portfolio inputs for optimizer-facing workflows."""

    expected_returns: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(EXPECTED_RETURN_COLUMNS)
    )
    covariance: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(COVARIANCE_COLUMNS)
    )
    exposures: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(EXPOSURE_COLUMNS)
    )
    eligibility: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(ELIGIBILITY_COLUMNS)
    )

    @classmethod
    def empty(cls) -> PortfolioInputBundle:
        return cls()

    @classmethod
    def from_frames(
        cls,
        *,
        expected_returns: pd.DataFrame | None = None,
        covariance: pd.DataFrame | None = None,
        exposures: pd.DataFrame | None = None,
        eligibility: pd.DataFrame | None = None,
    ) -> PortfolioInputBundle:
        return cls(
            expected_returns=_normalize_frame(
                expected_returns, columns=EXPECTED_RETURN_COLUMNS
            ),
            covariance=_normalize_frame(covariance, columns=COVARIANCE_COLUMNS),
            exposures=_normalize_frame(exposures, columns=EXPOSURE_COLUMNS),
            eligibility=_normalize_frame(eligibility, columns=ELIGIBILITY_COLUMNS),
        )
