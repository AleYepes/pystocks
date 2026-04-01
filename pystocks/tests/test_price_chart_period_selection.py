from datetime import date

from pystocks.fundamentals import FundamentalScraper


class _StoreStub:
    def __init__(self, latest):
        self.latest = latest

    def get_latest_price_series_effective_at(self, _conid):
        return self.latest


def _scraper_with_latest(latest):
    scraper = FundamentalScraper.__new__(FundamentalScraper)
    scraper.store = _StoreStub(latest)
    return scraper


def test_select_price_chart_period_uses_max_when_no_existing_series():
    scraper = _scraper_with_latest(None)
    assert (
        scraper._select_price_chart_period("1001", as_of_date=date(2026, 2, 27))
        == "MAX"
    )


def test_select_price_chart_period_avoids_day_periods_for_small_gaps():
    scraper = _scraper_with_latest(date(2026, 2, 26))
    assert (
        scraper._select_price_chart_period("1002", as_of_date=date(2026, 2, 27)) == "1W"
    )


def test_select_price_chart_period_scales_by_missing_window():
    as_of = date(2026, 2, 27)
    cases = [
        (date(2026, 2, 7), "1M"),
        (date(2025, 12, 9), "3M"),
        (date(2025, 9, 10), "6M"),
        (date(2025, 3, 14), "1Y"),
        (date(2023, 12, 20), "3Y"),
        (date(2021, 7, 3), "5Y"),
        (date(2017, 11, 10), "10Y"),
        (date(2012, 11, 10), "MAX"),
    ]
    for latest, expected in cases:
        scraper = _scraper_with_latest(latest)
        assert scraper._select_price_chart_period("1003", as_of_date=as_of) == expected


def test_price_chart_endpoint_uses_selected_period():
    scraper = _scraper_with_latest(None)
    endpoint = scraper._build_price_chart_endpoint("89384980", chart_period="3M")
    assert endpoint == "mf_performance_chart/89384980?chart_period=3M"
