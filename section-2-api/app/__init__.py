import sys
from pathlib import Path

# Add the parent of `app/` to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

__version__ = "0.0.1"