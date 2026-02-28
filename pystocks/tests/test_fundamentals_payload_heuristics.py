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


def test_ratios_payload_is_useful_for_bond_style_fixed_income_response():
    scraper = _scraper()
    payload = {
        "as_of_date": 1769835600000,
        "ratios": [],
        "financials": [],
        "fixed_income": [
            {"name": "Yield to Maturity", "value": 4.4559721868},
        ],
        "dividend": [],
        "zscore": [],
    }
    assert scraper._has_payload_data(payload, "ratios")


def test_ratios_payload_is_useful_when_only_as_of_date_is_present():
    scraper = _scraper()
    payload = {
        "as_of_date": 1769835600000,
        "ratios": [],
        "financials": [],
        "fixed_income": [],
        "dividend": [],
        "zscore": [],
    }
    assert scraper._has_payload_data(payload, "ratios")


def test_esg_endpoint_reloads_account_id_from_session_state():
    class DummySession:
        def __init__(self):
            self.account = None

        def get_primary_account_id(self):
            return self.account

    scraper = _scraper()
    scraper.session = DummySession()
    scraper.esg_account_id = None

    endpoint_without_account = scraper._build_esg_endpoint("564156940")
    assert endpoint_without_account == "impact/esg/564156940"

    scraper.session.account = "U19746488"
    endpoint_with_account = scraper._build_esg_endpoint("564156940")
    assert endpoint_with_account == "impact/esg/564156940?accounts=U19746488"
