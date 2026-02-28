import json
import tempfile
from pathlib import Path

from pystocks.session import IBKRSession


def test_get_login_credentials_from_config_file():
    tmp = tempfile.TemporaryDirectory()
    try:
        state_path = Path(tmp.name) / "auth_state.json"
        credentials_path = Path(tmp.name) / "login_credentials.json"
        credentials_path.write_text(json.dumps({"username": "demo_user", "password": "demo_pass"}))

        session = IBKRSession(state_path=state_path, credentials_path=credentials_path)
        username, password = session._get_login_credentials(prompt_if_missing=False)
        assert username == "demo_user"
        assert password == "demo_pass"
    finally:
        tmp.cleanup()


def test_get_login_credentials_prompts_and_writes_config(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    try:
        state_path = Path(tmp.name) / "auth_state.json"
        credentials_path = Path(tmp.name) / "login_credentials.json"
        session = IBKRSession(state_path=state_path, credentials_path=credentials_path)

        monkeypatch.setattr("builtins.input", lambda _: "prompt_user")
        monkeypatch.setattr("getpass.getpass", lambda _: "prompt_pass")

        username, password = session._get_login_credentials(prompt_if_missing=True)
        assert username == "prompt_user"
        assert password == "prompt_pass"
        saved = json.loads(credentials_path.read_text())
        assert saved == {"username": "prompt_user", "password": "prompt_pass"}
    finally:
        tmp.cleanup()


def test_get_login_credentials_force_prompt_overrides_saved_config(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    try:
        state_path = Path(tmp.name) / "auth_state.json"
        credentials_path = Path(tmp.name) / "login_credentials.json"
        credentials_path.write_text(json.dumps({"username": "saved_user", "password": "saved_pass"}))
        session = IBKRSession(state_path=state_path, credentials_path=credentials_path)

        monkeypatch.setattr("builtins.input", lambda _: "new_user")
        monkeypatch.setattr("getpass.getpass", lambda _: "new_pass")

        username, password = session._get_login_credentials(
            prompt_if_missing=True,
            force_prompt=True,
        )
        assert username == "new_user"
        assert password == "new_pass"
        saved = json.loads(credentials_path.read_text())
        assert saved == {"username": "new_user", "password": "new_pass"}
    finally:
        tmp.cleanup()
