from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pystocks_next.collection.session import CollectionSession


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload))


def test_primary_account_prefers_acesws_cookie_path(tmp_path: Path) -> None:
    state_path = tmp_path / "auth_state.json"
    _write_state(
        state_path,
        {
            "cookies": [
                {"path": "/portal.proxy/v1/portal/portfolio2/U10000000"},
                {"path": "/portal.proxy/v1/portal/acesws/U20000000"},
            ]
        },
    )
    session = CollectionSession(state_path=state_path)

    assert session.get_primary_account_id() == "U20000000"


def test_primary_account_prefers_persisted_state_field(tmp_path: Path) -> None:
    state_path = tmp_path / "auth_state.json"
    _write_state(
        state_path,
        {
            "primary_account_id": "U19746488",
            "cookies": [
                {"path": "/portal.proxy/v1/portal/portfolio2/U30000000"},
            ],
        },
    )
    session = CollectionSession(state_path=state_path)

    assert session.get_primary_account_id() == "U19746488"


def test_extract_primary_account_id_from_one_bar_payload() -> None:
    session = CollectionSession(state_path=Path("/tmp/unused.json"))

    payload = {
        "mostRelevantAccount": "U19746488",
        "portfolioAccounts": [
            {"accountId": "U30000000", "accountVan": "U30000000"},
        ],
    }

    assert (
        session._extract_primary_account_id_from_one_bar_payload(payload) == "U19746488"
    )


def test_validate_auth_state_persists_primary_account_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "auth_state.json"
    _write_state(state_path, {"cookies": [{"name": "x-sess-uuid", "value": "demo"}]})
    session = CollectionSession(state_path=state_path)

    async def fake_validate_state(
        _state: dict[str, object],
        *,
        timeout_s: float = 20.0,
    ) -> bool:
        assert timeout_s == 20.0
        return True

    async def fake_fetch_account_id(
        _state: dict[str, object],
        *,
        timeout_s: float = 20.0,
    ) -> str:
        assert timeout_s == 20.0
        return "U19746488"

    monkeypatch.setattr(session, "_validate_state_payload", fake_validate_state)
    monkeypatch.setattr(
        session, "_fetch_primary_account_id_from_state", fake_fetch_account_id
    )

    assert asyncio.run(session.validate_auth_state())
    saved = json.loads(state_path.read_text())
    assert saved["primary_account_id"] == "U19746488"


def test_session_exposes_fundamentals_auth_probe() -> None:
    session = CollectionSession(state_path=Path("/tmp/unused.json"))

    assert session.auth_check_endpoints == (
        "/tws.proxy/fundamentals/landing/756733?widgets=objective",
    )


def test_default_state_path_matches_legacy_auth_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(Path("/tmp"))

    session = CollectionSession()

    assert session.state_path == Path("/tmp/data/auth_state.json")
    assert session.credentials_path == Path("/tmp/data/login_credentials.json")
