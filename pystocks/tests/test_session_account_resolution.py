import json
import tempfile
from pathlib import Path

from pystocks.session import IBKRSession


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
