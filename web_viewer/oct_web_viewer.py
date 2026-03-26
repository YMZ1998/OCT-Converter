from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_viewer.oct.viewer import main

#netstat -ano | findstr :8765
# taskkill /PID 12345 /F
if __name__ == "__main__":
    main()
