from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RESEARCH_DIR = DATA_DIR / "research"

SESSION_STATE_PATH = DATA_DIR / "auth_state.json"
SQLITE_DB_PATH = DATA_DIR / "pystocks.sqlite"

for d in [DATA_DIR, RESEARCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)
