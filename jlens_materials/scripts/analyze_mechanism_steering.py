#!/usr/bin/env python3
"""Analyze an archived mechanism-steering study through a neutral CLI."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a frozen materials mechanism-steering experiment."
    )
    parser.add_argument(
        "--study", choices=("broad-screen", "prospective-grain"), required=True
    )
    args = parser.parse_args()

    if args.study == "broad-screen":
        from analyze_semantic_steering_v3 import analyze
    else:
        from analyze_relational_grain_steering_v4 import main as analyze
    analyze()


if __name__ == "__main__":
    main()
