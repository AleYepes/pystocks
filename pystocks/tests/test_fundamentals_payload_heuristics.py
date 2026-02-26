from pystocks.fundamentals import FundamentalScraper


def _scraper():
    return FundamentalScraper.__new__(FundamentalScraper)


def test_holdings_payload_is_useful_when_only_currency_is_present():
    scraper = _scraper()
    payload = {
        "currency": [{"name": "US Dollar", "weight": 100.0}],
    }
    assert scraper._has_payload_data(payload, "holdings")


def test_holdings_payload_is_useful_when_only_as_of_date_is_present():
    scraper = _scraper()
    payload = {
        "as_of_date": "2026-02-20",
    }
    assert scraper._has_payload_data(payload, "holdings")
