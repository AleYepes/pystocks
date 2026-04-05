import pandas as pd

from ..config import SQLITE_DB_PATH
from .txn import StorageTransaction, transaction


def query_frame(
    query: str,
    *,
    sqlite_path=SQLITE_DB_PATH,
    params=None,
    tx: StorageTransaction | None = None,
) -> pd.DataFrame:
    if tx is not None:
        return pd.read_sql_query(query, tx.connection, params=params)
    with transaction(sqlite_path) as managed_tx:
        frame = pd.read_sql_query(query, managed_tx.connection, params=params)
    return frame


def load_price_history(
    sqlite_path=SQLITE_DB_PATH, *, tx: StorageTransaction | None = None
) -> pd.DataFrame:
    df = query_frame(
        """
        SELECT
            conid,
            effective_at AS trade_date,
            price,
            open,
            high,
            low,
            close
        FROM price_chart_series
        ORDER BY conid, effective_at
        """,
        sqlite_path=sqlite_path,
        tx=tx,
    )
    if df.empty:
        return df

    df["conid"] = df["conid"].astype(str)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ["price", "open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_dividend_events(
    sqlite_path=SQLITE_DB_PATH, *, tx: StorageTransaction | None = None
) -> pd.DataFrame:
    df = query_frame(
        """
        SELECT
            d.conid,
            p.symbol,
            d.effective_at AS event_date,
            d.amount,
            d.currency AS dividend_currency,
            p.currency AS product_currency,
            d.description,
            d.event_type,
            d.declaration_date,
            d.record_date,
            d.payment_date
        FROM dividends_events_series d
        LEFT JOIN products p
          ON p.conid = d.conid
        ORDER BY d.conid, d.effective_at
        """,
        sqlite_path=sqlite_path,
        tx=tx,
    )
    if df.empty:
        return df

    df["conid"] = df["conid"].astype(str)
    df["event_date"] = pd.to_datetime(df["event_date"])
    for col in ["declaration_date", "record_date", "payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def load_risk_free_daily(
    sqlite_path=SQLITE_DB_PATH, *, tx: StorageTransaction | None = None
) -> pd.DataFrame:
    df = query_frame(
        """
        SELECT
            trade_date,
            nominal_rate,
            daily_nominal_rate,
            source_count,
            observed_at
        FROM supplementary_risk_free_daily
        ORDER BY trade_date
        """,
        sqlite_path=sqlite_path,
        tx=tx,
    )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    for col in ["nominal_rate", "daily_nominal_rate", "source_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_world_bank_country_features(
    sqlite_path=SQLITE_DB_PATH, *, tx: StorageTransaction | None = None
) -> pd.DataFrame:
    df = query_frame(
        """
        SELECT
            economy_code,
            effective_at,
            feature_year,
            population_level,
            population_growth,
            gdp_pcap_level,
            gdp_pcap_growth,
            economic_output_gdp_level,
            economic_output_gdp_growth,
            foreign_direct_investment_level,
            foreign_direct_investment_growth,
            share_trade_volume_level,
            share_trade_volume_growth,
            observed_at
        FROM supplementary_world_bank_country_features
        ORDER BY economy_code, effective_at
        """,
        sqlite_path=sqlite_path,
        tx=tx,
    )
    if df.empty:
        return df
    df["economy_code"] = df["economy_code"].astype(str).str.upper()
    df["effective_at"] = pd.to_datetime(df["effective_at"])
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    for col in df.columns:
        if col in {"economy_code", "effective_at", "observed_at"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_snapshot_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "conid" in out.columns:
        out["conid"] = out["conid"].astype(str)
    if "effective_at" in out.columns:
        out["effective_at"] = pd.to_datetime(out["effective_at"])
    return out


def _load_snapshot_frame(
    query: str,
    *,
    sqlite_path=SQLITE_DB_PATH,
    tx: StorageTransaction | None = None,
) -> pd.DataFrame:
    return _normalize_snapshot_frame(query_frame(query, sqlite_path=sqlite_path, tx=tx))


def load_snapshot_feature_tables(
    sqlite_path=SQLITE_DB_PATH, *, tx: StorageTransaction | None = None
) -> dict[str, pd.DataFrame]:
    return {
        "profile_and_fees": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                asset_type,
                classification,
                distribution_details,
                domicile,
                fiscal_date,
                fund_category,
                fund_management_company,
                fund_manager_benchmark,
                fund_market_cap_focus,
                geographical_focus,
                inception_date,
                management_approach,
                management_expenses,
                manager_tenure,
                maturity_date,
                objective_type,
                portfolio_manager,
                redemption_charge_actual,
                redemption_charge_max,
                scheme,
                total_expense_ratio,
                total_net_assets_value,
                total_net_assets_date,
                objective,
                jap_fund_warning,
                theme_name
            FROM profile_and_fees
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_asset_type": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                equity,
                cash,
                fixed_income,
                other
            FROM holdings_asset_type
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_debtor_quality": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                quality_aaa,
                quality_aa,
                quality_a,
                quality_bbb,
                quality_bb,
                quality_b,
                quality_ccc,
                quality_cc,
                quality_c,
                quality_d,
                quality_not_rated,
                quality_not_available
            FROM holdings_debtor_quality
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_maturity": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                maturity_less_than_1_year,
                maturity_1_to_3_years,
                maturity_3_to_5_years,
                maturity_5_to_10_years,
                maturity_10_to_20_years,
                maturity_20_to_30_years,
                maturity_greater_than_30_years,
                maturity_other
            FROM holdings_maturity
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_industry": _load_snapshot_frame(
            "SELECT conid, effective_at, industry, value_num FROM holdings_industry",
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_currency": _load_snapshot_frame(
            """
            SELECT conid, effective_at, COALESCE(code, currency) AS code, currency, value_num
            FROM holdings_currency
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_investor_country": _load_snapshot_frame(
            """
            SELECT conid, effective_at, COALESCE(country_code, country) AS country_code, country, value_num
            FROM holdings_investor_country
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_geographic_weights": _load_snapshot_frame(
            "SELECT conid, effective_at, region, value_num FROM holdings_geographic_weights",
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_debt_type": _load_snapshot_frame(
            "SELECT conid, effective_at, debt_type, value_num FROM holdings_debt_type",
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "holdings_top10": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                name,
                holding_weight_num
            FROM holdings_top10
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "ratios_key_ratios": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_key_ratios
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "ratios_financials": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_financials
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "ratios_fixed_income": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_fixed_income
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "ratios_dividend": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_dividend
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "ratios_zscore": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_zscore
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "dividends_industry_metrics": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                dividend_yield,
                annual_dividend,
                dividend_ttm,
                dividend_yield_ttm,
                currency
            FROM dividends_industry_metrics
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "morningstar_summary": _load_snapshot_frame(
            """
            SELECT
                conid,
                effective_at,
                medalist_rating,
                process,
                people,
                parent,
                morningstar_rating,
                sustainability_rating,
                category,
                category_index
            FROM morningstar_summary
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
        "lipper_ratings": _load_snapshot_frame(
            """
            SELECT conid, effective_at, period, metric_id, rating_value AS value_num
            FROM lipper_ratings
            """,
            sqlite_path=sqlite_path,
            tx=tx,
        ),
    }
