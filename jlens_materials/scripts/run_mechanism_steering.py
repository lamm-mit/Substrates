#!/usr/bin/env python3
"""Run an archived mechanism-steering study through a neutral CLI."""

from __future__ import annotations

import argparse
import sys

from run_semantic_steering_v3 import main as run_manifest


STUDIES = {
    "broad-screen": (
        "experiments/semantic-steering-v3-preregistration.json",
        "experiments/semantic-steering-v3_raw.json",
    ),
    "prospective-grain": (
        "experiments/relational-grain-steering-v4-preregistration.json",
        "experiments/relational-grain-steering-v4_raw.json",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen materials mechanism-steering experiment."
    )
    parser.add_argument("--study", choices=sorted(STUDIES), required=True)
    args, forwarded = parser.parse_known_args()
    manifest, output = STUDIES[args.study]
    sys.argv = [
        "run_mechanism_steering.py",
        "--manifest",
        manifest,
        "--output",
        output,
        *forwarded,
    ]
    run_manifest()


if __name__ == "__main__":
    main()
