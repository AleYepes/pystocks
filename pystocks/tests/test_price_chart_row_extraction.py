from pystocks.fundamentals_store import _extract_price_chart_rows


def _payload(points):
    return {
        "plot": {
            "series": [
                {
                    "name": "price",
                    "plotData": points,
                }
            ]
        }
    }


def test_price_chart_row_extraction_prefers_x_date_trade_date():
    rows = _extract_price_chart_rows(
        _payload(
            [
                {"x": 1319158800000, "debugY": 20111020, "y": 1.0},
            ]
        )
    )
    assert len(rows) == 1
    assert rows[0]["effective_at"] == "2011-10-21"
    assert rows[0]["debug_mismatch"] == 0


def test_price_chart_row_extraction_flags_only_large_date_divergence():
    rows = _extract_price_chart_rows(
        _payload(
            [
                {"x": 1700000000000, "debugY": 20231114, "y": 1.0},
                {"x": 1700000000000, "debugY": 20231110, "y": 2.0},
            ]
        )
    )
    assert len(rows) == 2
    assert rows[0]["debug_mismatch"] == 0
    assert rows[1]["debug_mismatch"] == 1
