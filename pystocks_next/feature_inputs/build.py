from __future__ import annotations

import sqlite3

from ..progress import ProgressSink
from .bundle import AnalysisInputBundle
from .dividends import DividendInputConfig, build_dividend_input_bundle
from .prices import PriceInputConfig, build_price_input_bundle
from .snapshots import SnapshotInputConfig, build_snapshot_input_bundle
from .supplementary import SupplementaryInputConfig, build_supplementary_input_bundle


def build_analysis_input_bundle(
    *,
    conn: sqlite3.Connection,
    price_config: PriceInputConfig | None = None,
    dividend_config: DividendInputConfig | None = None,
    snapshot_config: SnapshotInputConfig | None = None,
    supplementary_config: SupplementaryInputConfig | None = None,
    progress: ProgressSink | None = None,
) -> AnalysisInputBundle:
    tracker = (
        progress.stage("Building feature inputs", total=4, unit="step")
        if progress is not None
        else None
    )
    price_bundle = build_price_input_bundle(conn=conn, config=price_config)
    if tracker is not None:
        tracker.advance(detail="Built price inputs")
    dividend_bundle = build_dividend_input_bundle(
        conn=conn,
        price_reference=price_bundle,
        config=dividend_config,
    )
    if tracker is not None:
        tracker.advance(detail="Built dividend inputs")
    snapshot_bundle = build_snapshot_input_bundle(conn=conn, config=snapshot_config)
    if tracker is not None:
        tracker.advance(detail="Built snapshot inputs")
    supplementary_bundle = build_supplementary_input_bundle(
        conn=conn,
        config=supplementary_config,
    )
    if tracker is not None:
        tracker.advance(detail="Built supplementary inputs")
        tracker.close(detail="Feature inputs ready")
    return AnalysisInputBundle.from_frames(
        prices=price_bundle.prices,
        price_eligibility=price_bundle.price_eligibility,
        dividends=dividend_bundle.dividends,
        dividend_summary=dividend_bundle.dividend_summary,
        snapshot_features=snapshot_bundle.snapshot_features,
        snapshot_holdings_diagnostics=snapshot_bundle.snapshot_holdings_diagnostics,
        snapshot_ratio_diagnostics=snapshot_bundle.snapshot_ratio_diagnostics,
        snapshot_table_summary=snapshot_bundle.snapshot_table_summary,
        risk_free_daily=supplementary_bundle.risk_free_daily,
        macro_features=supplementary_bundle.macro_features,
    )
