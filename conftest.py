import sys
from pathlib import Path

# Ensure project root is on sys.path so `from app.xxx` works in tests
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "eval"))
