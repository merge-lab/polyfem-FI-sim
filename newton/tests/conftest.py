import sys
from pathlib import Path

# Scripts in newton/ import each other flat (e.g. `from utils_FI import ...`),
# so the tests need newton/ itself on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
