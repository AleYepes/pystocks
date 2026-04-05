import asyncio
import json
import tempfile
from pathlib import Path

from pystocks.ingest.session import IBKRSession


def _write_state(payload):
    tmp = tempfile.TemporaryDirectory()
    state_path = Path(tmp.name) / "auth_state.json"
    state_path.write_text(json.dumps(payload))
    return tmp, state_path


def test_primary_account_prefers_acesws_cookie_path():
    payload = {
        "cookies": [
            {"path": "/portal.proxy/v1/portal/portfolio2/U10000000"},
            {"path": "/portal.proxy/v1/portal/acesws/U20000000"},
        ]
    }
    tmp, state_path = _write_state(payload)
    try:
        session = IBKRSession(state_path=state_path)
        assert session.get_primary_account_id() == "U20000000"
    finally:
        tmp.cleanup()


def test_primary_account_falls_back_to_portfolio2_cookie_path():
    payload = {
        "cookies": [
            {"path": "/portal.proxy/v1/portal/portfolio2/U30000000"},
            {"path": "/portal.proxy/v1/portal/portfolio2/U40000000"},
        ]
    }
    tmp, state_path = _write_state(payload)
    try:
        session = IBKRSession(state_path=state_path)
        assert session.get_primary_account_id() == "U30000000"
    finally:
        tmp.cleanup()


def test_primary_account_prefers_persisted_state_field():
    payload = {
        "primary_account_id": "U19746488",
        "cookies": [
            {"path": "/portal.proxy/v1/portal/portfolio2/U30000000"},
        ],
    }
    tmp, state_path = _write_state(payload)
    try:
        session = IBKRSession(state_path=state_path)
        assert session.get_primary_account_id() == "U19746488"
    finally:
        tmp.cleanup()


def test_extract_primary_account_id_from_one_bar_payload():
    session = IBKRSession()
    payload = {
        "mostRelevantAccount": "U19746488",
        "portfolioAccounts": [
            {
                "accountId": "U30000000",
                "accountVan": "U30000000",
            }
        ],
    }
    assert (
        session._extract_primary_account_id_from_one_bar_payload(payload) == "U19746488"
    )


def test_validate_auth_state_persists_primary_account_id(monkeypatch):
    payload = {"cookies": [{"name": "x-sess-uuid", "value": "demo"}]}
    tmp, state_path = _write_state(payload)
    try:
        session = IBKRSession(state_path=state_path)

        async def fake_validate_state(state, timeout_s=20.0):
            return True

        async def fake_fetch_account_id(state, timeout_s=20.0):
            return "U19746488"

        monkeypatch.setattr(session, "_validate_state_payload", fake_validate_state)
        monkeypatch.setattr(
            session, "_fetch_primary_account_id_from_state", fake_fetch_account_id
        )

        assert asyncio.run(session.validate_auth_state())
        saved = json.loads(state_path.read_text())
        assert saved["primary_account_id"] == "U19746488"
    finally:
        tmp.cleanup()


def test_auth_validation_uses_fundamentals_probe_only():
    session = IBKRSession()
    assert session._auth_check_endpoints == [
        "/tws.proxy/fundamentals/landing/756733?widgets=objective"
    ]
