import hashlib
import json
import math
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd

from .config import DATA_DIR, RESEARCH_DIR, SQLITE_DB_PATH

PRICE_FIELDS = ("price", "open", "high", "low", "close")
DEFAULT_OFFICIAL_DB_PATH = str(DATA_DIR / "pystocks_official_prices.sqlite")
DEFAULT_COMPARISON_OUT_PATH = str(RESEARCH_DIR / "official_vs_fundamentals_price_comparison_latest.csv")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        out = float(s)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _to_iso_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if len(s) >= 10:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
            except Exception:
                pass
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
                try:
                    return datetime.strptime(s[:10], fmt).date().isoformat()
                except Exception:
                    continue
        if s.isdigit() and len(s) == 8:
            try:
                return datetime.strptime(s, "%Y%m%d").date().isoformat()
            except Exception:
                return None
    return None


def _load_conids_from_file(path):
    if path is None:
        return []
    lines = Path(path).read_text().splitlines()
    out = []
    seen = set()
    for line in lines:
        value = str(line).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _derive_summary_path(path):
    base = Path(path)
    return str(base.with_name(f"{base.stem}_summary{base.suffix or '.csv'}"))


def _normalize_bars(bars):
    raw_rows = []
    normalized_rows = []
    for bar in bars or []:
        bar_date = _to_iso_date(getattr(bar, "date", None))
        row = {
            "effective_at": bar_date,
            "price": _to_float(getattr(bar, "close", None)),
            "open": _to_float(getattr(bar, "open", None)),
            "high": _to_float(getattr(bar, "high", None)),
            "low": _to_float(getattr(bar, "low", None)),
            "close": _to_float(getattr(bar, "close", None)),
        }
        if row["effective_at"] is None:
            continue
        if not any(row.get(field) is not None for field in PRICE_FIELDS):
            continue
        raw_rows.append(
            {
                "date": str(getattr(bar, "date", "")),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": _to_float(getattr(bar, "volume", None)),
                "average": _to_float(getattr(bar, "average", None)),
                "bar_count": _to_float(getattr(bar, "barCount", None)),
            }
        )
        normalized_rows.append(row)
    return raw_rows, normalized_rows


class OfficialPriceStore:
    def __init__(self, sqlite_path=DEFAULT_OFFICIAL_DB_PATH):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._compressor = zstd.ZstdCompressor(level=10)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_payload_blobs (
                    payload_hash TEXT PRIMARY KEY,
                    compression TEXT NOT NULL,
                    raw_size_bytes INTEGER NOT NULL,
                    compressed_size_bytes INTEGER NOT NULL,
                    payload_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS price_chart_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    points_count INTEGER,
                    min_trade_date TEXT,
                    max_trade_date TEXT,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS price_chart_series (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    price REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    PRIMARY KEY (conid, effective_at)
                );

                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    bars_received INTEGER NOT NULL,
                    payload_hash TEXT,
                    duration_str TEXT NOT NULL,
                    bar_size_setting TEXT NOT NULL,
                    what_to_show TEXT NOT NULL,
                    use_rth INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_official_price_snapshots_hash ON price_chart_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_official_price_series_effective_at ON price_chart_series(effective_at);
                CREATE INDEX IF NOT EXISTS idx_official_fetch_log_conid ON fetch_log(conid);
                """
            )

    def _store_blob(self, conn, payload):
        raw_bytes = _canonical_json_bytes(payload)
        payload_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing = conn.execute(
            """
            SELECT payload_hash
            FROM raw_payload_blobs
            WHERE payload_hash = ?
            """,
            [payload_hash],
        ).fetchone()

        if existing is None:
            compressed = self._compressor.compress(raw_bytes)
            conn.execute(
                """
                INSERT INTO raw_payload_blobs (
                    payload_hash,
                    compression,
                    raw_size_bytes,
                    compressed_size_bytes,
                    payload_blob,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    payload_hash,
                    "zstd",
                    len(raw_bytes),
                    len(compressed),
                    compressed,
                    _now_iso(),
                ],
            )
        return payload_hash

    def persist_price_data(self, conid, observed_at, request_payload, raw_rows, rows):
        if not rows:
            return {"payload_hash": None, "rows_written": 0}

        dates = sorted([r["effective_at"] for r in rows if r.get("effective_at")])
        if not dates:
            return {"payload_hash": None, "rows_written": 0}

        payload = {
            "conid": str(conid),
            "observed_at": str(observed_at),
            "request": dict(request_payload or {}),
            "bars": list(raw_rows or []),
        }
        now_iso = _now_iso()
        min_date = dates[0]
        max_date = dates[-1]

        with self._get_conn() as conn:
            payload_hash = self._store_blob(conn, payload)
            conn.execute(
                """
                INSERT INTO price_chart_snapshots (
                    conid,
                    effective_at,
                    observed_at,
                    payload_hash,
                    inserted_at,
                    updated_at,
                    points_count,
                    min_trade_date,
                    max_trade_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conid, effective_at) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    payload_hash = excluded.payload_hash,
                    updated_at = excluded.updated_at,
                    points_count = excluded.points_count,
                    min_trade_date = excluded.min_trade_date,
                    max_trade_date = excluded.max_trade_date
                """,
                [
                    str(conid),
                    str(max_date),
                    str(observed_at),
                    str(payload_hash),
                    now_iso,
                    now_iso,
                    len(rows),
                    str(min_date),
                    str(max_date),
                ],
            )
            conn.executemany(
                """
                INSERT INTO price_chart_series (
                    conid,
                    effective_at,
                    price,
                    open,
                    high,
                    low,
                    close
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conid, effective_at) DO UPDATE SET
                    price = excluded.price,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close
                """,
                [
                    [
                        str(conid),
                        str(row.get("effective_at")),
                        _to_float(row.get("price")),
                        _to_float(row.get("open")),
                        _to_float(row.get("high")),
                        _to_float(row.get("low")),
                        _to_float(row.get("close")),
                    ]
                    for row in rows
                ],
            )
            conn.commit()

        return {"payload_hash": payload_hash, "rows_written": len(rows)}

    def log_fetch(
        self,
        conid,
        requested_at,
        finished_at,
        status,
        error,
        bars_received,
        payload_hash,
        duration_str,
        bar_size_setting,
        what_to_show,
        use_rth,
    ):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO fetch_log (
                    conid,
                    requested_at,
                    finished_at,
                    status,
                    error,
                    bars_received,
                    payload_hash,
                    duration_str,
                    bar_size_setting,
                    what_to_show,
                    use_rth
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(conid),
                    str(requested_at),
                    str(finished_at),
                    str(status),
                    str(error) if error is not None else None,
                    int(bars_received or 0),
                    str(payload_hash) if payload_hash is not None else None,
                    str(duration_str),
                    str(bar_size_setting),
                    str(what_to_show),
                    1 if use_rth else 0,
                ],
            )
            conn.commit()


def _load_series_df(sqlite_path, conids=None):
    conids = list(conids or [])
    try:
        with sqlite3.connect(str(sqlite_path)) as conn:
            if conids:
                placeholders = ", ".join(["?"] * len(conids))
                return pd.read_sql_query(
                    f"""
                    SELECT conid, effective_at, price, open, high, low, close
                    FROM price_chart_series
                    WHERE conid IN ({placeholders})
                    """,
                    conn,
                    params=[str(c) for c in conids],
                )
            return pd.read_sql_query(
                """
                SELECT conid, effective_at, price, open, high, low, close
                FROM price_chart_series
                """,
                conn,
            )
    except sqlite3.OperationalError:
        return pd.DataFrame(columns=["conid", "effective_at", "price", "open", "high", "low", "close"])


def build_comparison_frames(official_df, fundamentals_df):
    official = official_df.copy()
    fundamentals = fundamentals_df.copy()

    if official.empty or fundamentals.empty:
        return pd.DataFrame(), pd.DataFrame()

    official = official.rename(
        columns={field: f"{field}_official" for field in PRICE_FIELDS}
    )
    fundamentals = fundamentals.rename(
        columns={field: f"{field}_fundamentals" for field in PRICE_FIELDS}
    )

    detail = official.merge(fundamentals, on=["conid", "effective_at"], how="inner")
    if detail.empty:
        return detail, pd.DataFrame()

    for field in PRICE_FIELDS:
        off_col = f"{field}_official"
        fund_col = f"{field}_fundamentals"
        diff_col = f"{field}_diff"
        abs_col = f"{field}_diff_abs"
        pct_col = f"{field}_diff_pct"

        detail[diff_col] = detail[off_col] - detail[fund_col]
        detail[abs_col] = detail[diff_col].abs()
        denom = detail[fund_col].abs()
        detail[pct_col] = np.where(denom > 0, detail[diff_col] / detail[fund_col], np.nan)

    detail = detail.sort_values(["conid", "effective_at"]).reset_index(drop=True)

    grouped = detail.groupby("conid", dropna=False)
    summary = grouped.agg(
        overlap_rows=("effective_at", "count"),
        first_effective_at=("effective_at", "min"),
        last_effective_at=("effective_at", "max"),
    ).reset_index()

    for field in PRICE_FIELDS:
        abs_col = f"{field}_diff_abs"
        summary[f"{field}_mean_abs_diff"] = grouped[abs_col].mean().values
        summary[f"{field}_median_abs_diff"] = grouped[abs_col].median().values
        summary[f"{field}_max_abs_diff"] = grouped[abs_col].max().values

    close_pct = grouped["close_diff_pct"]
    summary["close_pct_rows_abs_diff_gt_1pct"] = close_pct.apply(
        lambda s: float((s.abs() > 0.01).mean())
    ).values
    summary["close_pct_rows_abs_diff_gt_3pct"] = close_pct.apply(
        lambda s: float((s.abs() > 0.03).mean())
    ).values
    summary = summary.sort_values(
        ["close_mean_abs_diff", "overlap_rows"], ascending=[False, False]
    ).reset_index(drop=True)

    return detail, summary


def _resolve_contract(ib, conid):
    from ib_async import Contract

    candidates = [
        Contract(conId=int(conid), exchange="SMART"),
        Contract(conId=int(conid), secType="STK", exchange="SMART"),
    ]
    for contract in candidates:
        details = ib.reqContractDetails(contract)
        if details:
            return details[0].contract
    return None


def _fetch_bars_for_conid(ib, conid, duration, bar_size, what_to_show, use_rth, strict_contract):
    contract = _resolve_contract(ib, conid)
    if contract is None:
        if strict_contract:
            raise RuntimeError(f"No contract details found for conid={conid}")
        return [], []

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=str(duration),
        barSizeSetting=str(bar_size),
        whatToShow=str(what_to_show),
        useRTH=bool(use_rth),
    )
    return _normalize_bars(bars)


def run(
    conids_file="docs/sample_conids.txt",
    official_db=DEFAULT_OFFICIAL_DB_PATH,
    fundamentals_db=str(SQLITE_DB_PATH),
    host="127.0.0.1",
    port=7497,
    client_id=7,
    duration="10 Y",
    bar_size="1 day",
    what_to_show="TRADES",
    use_rth=True,
    sleep_ms=100,
    comparison_out=DEFAULT_COMPARISON_OUT_PATH,
    strict_contract=False,
):
    conids = _load_conids_from_file(conids_file)
    if not conids:
        raise ValueError(f"No conids found in file: {conids_file}")

    store = OfficialPriceStore(sqlite_path=official_db)
    fetched_conids = []
    status_counts = {"ok": 0, "empty": 0, "error": 0}

    from ib_async import IB

    ib = IB()
    connected = False
    try:
        ib.connect(str(host), int(port), clientId=int(client_id))
        connected = True

        for conid in conids:
            requested_at = _now_iso()
            finished_at = requested_at
            status = "error"
            error = None
            bars_received = 0
            payload_hash = None
            try:
                raw_rows, rows = _fetch_bars_for_conid(
                    ib=ib,
                    conid=conid,
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                    strict_contract=bool(strict_contract),
                )
                bars_received = len(rows)
                result = store.persist_price_data(
                    conid=conid,
                    observed_at=requested_at,
                    request_payload={
                        "duration_str": str(duration),
                        "bar_size_setting": str(bar_size),
                        "what_to_show": str(what_to_show),
                        "use_rth": bool(use_rth),
                    },
                    raw_rows=raw_rows,
                    rows=rows,
                )
                payload_hash = result.get("payload_hash")
                status = "ok" if bars_received > 0 else "empty"
                if bars_received > 0:
                    fetched_conids.append(str(conid))
            except Exception as exc:
                status = "error"
                error = str(exc)
            finally:
                finished_at = _now_iso()
                store.log_fetch(
                    conid=conid,
                    requested_at=requested_at,
                    finished_at=finished_at,
                    status=status,
                    error=error,
                    bars_received=bars_received,
                    payload_hash=payload_hash,
                    duration_str=duration,
                    bar_size_setting=bar_size,
                    what_to_show=what_to_show,
                    use_rth=bool(use_rth),
                )
                status_counts[status] += 1
                if int(sleep_ms) > 0:
                    time.sleep(int(sleep_ms) / 1000.0)
    finally:
        if connected:
            ib.disconnect()

    compare_conids = fetched_conids if fetched_conids else conids
    official_df = _load_series_df(official_db, conids=compare_conids)
    fundamentals_df = _load_series_df(fundamentals_db, conids=compare_conids)
    detail_df, summary_df = build_comparison_frames(official_df, fundamentals_df)

    detail_path = Path(comparison_out)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(_derive_summary_path(comparison_out))
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    overlap_rows = int(len(detail_df))
    conid_overlap = int(detail_df["conid"].nunique()) if not detail_df.empty else 0

    print("Official price fetch complete.")
    print(f"targeted_conids: {len(conids)}")
    print(f"fetched_conids: {len(fetched_conids)}")
    print(f"status_ok: {status_counts['ok']}")
    print(f"status_empty: {status_counts['empty']}")
    print(f"status_error: {status_counts['error']}")
    print(f"overlap_rows: {overlap_rows}")
    print(f"overlap_conids: {conid_overlap}")
    print(f"official_db: {official_db}")
    print(f"comparison_detail_csv: {detail_path}")
    print(f"comparison_summary_csv: {summary_path}")

    return {
        "targeted_conids": len(conids),
        "fetched_conids": len(fetched_conids),
        "status_counts": dict(status_counts),
        "overlap_rows": overlap_rows,
        "overlap_conids": conid_overlap,
        "official_db": str(official_db),
        "comparison_detail_csv": str(detail_path),
        "comparison_summary_csv": str(summary_path),
    }


if __name__ == "__main__":
    import fire

    fire.Fire(run)
