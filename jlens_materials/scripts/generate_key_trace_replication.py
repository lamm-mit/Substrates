#!/usr/bin/env python3
"""Extract frozen positive and negative v2 cases for trajectory serialization."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "prompts" / "materials-paper-v2-preregistered.json"
OUTPUT = ROOT / "prompts" / "materials-key-concept-traces-v1.json"

SLUGS = [
    "paper-v2-assoc-notch-resistance-04",
    "paper-v2-assoc-ductile-02",
    "paper-v2-assoc-line-defect-motion-01",
    "paper-v2-assoc-rapid-transformation-04",
    "paper-v2-assoc-cleavage-03",
    "paper-v2-assoc-hot-air-surface-layer-01",
]


def main() -> None:
    source = json.loads(SOURCE.read_text())
    by_slug = {item["slug"]: item for item in source["prompts"]}
    missing = [slug for slug in SLUGS if slug not in by_slug]
    if missing:
        raise ValueError(f"missing frozen prompts: {missing}")
    payload = {
        "description": (
            "Exact replication subset from materials-paper-v2. Four strong "
            "cases and two prespecified weak/negative cases are rerun only to "
            "persist per-layer J/logit trajectories for auditable figures."
        ),
        "prompts": [by_slug[slug] for slug in SLUGS],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
