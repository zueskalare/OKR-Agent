import sys
from pathlib import Path

# Make `import AtomAgents` / `import backend` work regardless of pytest rootdir.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES = Path(__file__).parent / "fixtures"
