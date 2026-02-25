from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PricePreprocessConfig:
    min_history_days: int = 252
    max_missing_ratio: float = 0.30
    max_internal_gap_days: int = 20
    stale_run_max_days: int = 5
    outlier_z_threshold: float = 50.0


def _compute_internal_gap_days(dates):
    if len(dates) <= 1:
        return 0
    date_arr = np.array(pd.to_datetime(dates).date, dtype="datetime64[D]")
    prev_dates = date_arr[:-1]
    next_dates = date_arr[1:]
    gap_days = np.busday_count(prev_dates, next_dates) - 1
    if gap_days.size == 0:
        return 0
    gap_days = np.maximum(gap_days, 0)
    return int(gap_days.max())


def _compute_eligibility(df, config: PricePreprocessConfig):
    rows = []
    for conid, group in df.groupby("conid"):
        valid = group[group["is_clean_price"]].copy()

        total_rows = int(len(group))
        valid_rows = int(len(valid))

        if valid.empty:
            min_date = None
            max_date = None
            expected_days = 0
            missing_ratio = 1.0
            max_internal_gap_days = 0
        else:
            valid_dates = sorted(valid["trade_date"].dropna().unique())
            min_date = str(valid_dates[0]) if valid_dates else None
            max_date = str(valid_dates[-1]) if valid_dates else None
            expected_days = (
                int(len(pd.bdate_range(valid_dates[0], valid_dates[-1])))
                if len(valid_dates) > 0
                else 0
            )
            missing_ratio = (
                float(max(0.0, 1.0 - (valid_rows / expected_days)))
                if expected_days > 0
                else 1.0
            )
            max_internal_gap_days = _compute_internal_gap_days(valid_dates)

        reasons = []
        if valid_rows < config.min_history_days:
            reasons.append("Insufficient history")
        if missing_ratio > config.max_missing_ratio:
            reasons.append("Excessive missing ratio")
        if max_internal_gap_days > config.max_internal_gap_days:
            reasons.append("Large internal gap")

        eligible = len(reasons) == 0
        reason = "OK" if eligible else "; ".join(reasons)

        rows.append(
            {
                "conid": str(conid),
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "min_date": min_date,
                "max_date": max_date,
                "expected_business_days": expected_days,
                "missing_ratio": missing_ratio,
                "max_internal_gap_days": max_internal_gap_days,
                "eligible": bool(eligible),
                "eligibility_reason": reason,
            }
        )

    return pd.DataFrame(rows).sort_values("conid").reset_index(drop=True)


if __name__ == "__main__":
    print(run())
