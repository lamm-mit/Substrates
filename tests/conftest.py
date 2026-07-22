from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "jlens_materials"
VENDOR = MATERIALS / "_vendor_jlens"
for path in (str(MATERIALS), str(VENDOR)):
    if path not in sys.path:
        sys.path.insert(0, path)
