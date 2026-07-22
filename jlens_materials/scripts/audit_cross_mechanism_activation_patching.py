#!/usr/bin/env python3
"""Independently audit cross-mechanism activation-patching outputs."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "cross-mechanism-activation-patching-2026-07-18"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(values)))
    )
    null = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(np.abs(null) >= abs(values.mean()) - 1e-12))


def main() -> None:
    raw = json.loads((OUT / "raw.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    amendment = json.loads((OUT / "protocol-amendment-v1.json").read_text())
    statistics = json.loads((OUT / "statistics.json").read_text())
    patch = pd.DataFrame(raw["patch_rows"])
    checks: list[dict] = []

    def check(name: str, passed: bool, **details) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    check("patch row count", len(patch) == 1920, observed=len(patch))
    cardinality = (
        patch.groupby("receiver_prompt_id")["donor_prompt_id"].nunique()
    )
    check(
        "twenty cross-mechanism donors per receiver",
        len(cardinality) == 24 and set(cardinality) == {20},
        receiver_count=len(cardinality),
        donor_counts=sorted(set(int(value) for value in cardinality)),
    )
    check(
        "four frozen layers",
        sorted(patch["layer"].unique().tolist()) == [16, 24, 32, 37],
        observed=sorted(patch["layer"].unique().tolist()),
    )
    check(
        "no duplicate intervention",
        not patch[
            ["receiver_prompt_id", "donor_prompt_id", "layer"]
        ].duplicated().any(),
    )

    runner_path = ROOT / protocol["inputs"]["runner"]["path"]
    check(
        "runner fingerprint",
        sha256(runner_path) == protocol["inputs"]["runner"]["sha256"],
        actual=sha256(runner_path),
        recorded=protocol["inputs"]["runner"]["sha256"],
    )
    base_path = ROOT / amendment["base_protocol"]["path"]
    check(
        "base protocol fingerprint",
        sha256(base_path) == amendment["base_protocol"]["sha256"],
        actual=sha256(base_path),
        recorded=amendment["base_protocol"]["sha256"],
    )
    analysis_path = ROOT / amendment["analysis_implementation"]["path"]
    check(
        "analysis fingerprint",
        sha256(analysis_path)
        == amendment["analysis_implementation"]["sha256"],
        actual=sha256(analysis_path),
        recorded=amendment["analysis_implementation"]["sha256"],
    )

    ordered_rows = []
    families = sorted(patch["donor_family"].unique())
    for donor_family in families:
        for receiver_family in families:
            if donor_family == receiver_family:
                continue
            subset = patch[
                (patch["donor_family"] == donor_family)
                & (patch["receiver_family"] == receiver_family)
            ]
            positive = subset[
                subset["donor_outcome_sign"] > 0
            ]["patched_positive_minus_negative"].mean()
            negative = subset[
                subset["donor_outcome_sign"] < 0
            ]["patched_positive_minus_negative"].mean()
            numeric_increase = subset[
                subset["donor_numeric_sign"] > 0
            ]["patched_positive_minus_negative"].mean()
            numeric_decrease = subset[
                subset["donor_numeric_sign"] < 0
            ]["patched_positive_minus_negative"].mean()
            pair = " | ".join(sorted([donor_family, receiver_family]))
            ordered_rows.append(
                {
                    "donor_family": donor_family,
                    "receiver_family": receiver_family,
                    "family_pair": pair,
                    "cross_vocabulary": bool(
                        subset["cross_vocabulary"].iloc[0]
                    ),
                    "opposite_orientation": bool(
                        subset["opposite_response_orientation"].iloc[0]
                    ),
                    "physical": float(positive - negative),
                    "numeric": float(numeric_increase - numeric_decrease),
                }
            )
    ordered = pd.DataFrame(ordered_rows)
    unordered = (
        ordered.groupby(
            [
                "family_pair",
                "cross_vocabulary",
                "opposite_orientation",
            ],
            as_index=False,
        )[["physical", "numeric"]]
        .mean()
    )
    stored_unordered = pd.read_csv(
        OUT / "unordered_mechanism_pair_effects.csv"
    ).sort_values("family_pair")
    unordered = unordered.sort_values("family_pair")
    check(
        "independent unordered physical effects",
        np.allclose(
            unordered["physical"],
            stored_unordered["physical_outcome_contrast"],
            atol=1e-12,
            rtol=1e-12,
        ),
        max_absolute_difference=float(
            np.max(
                np.abs(
                    unordered["physical"].to_numpy()
                    - stored_unordered[
                        "physical_outcome_contrast"
                    ].to_numpy()
                )
            )
        ),
    )
    check(
        "independent unordered numeric effects",
        np.allclose(
            unordered["numeric"],
            stored_unordered["numeric_direction_contrast"],
            atol=1e-12,
            rtol=1e-12,
        ),
        max_absolute_difference=float(
            np.max(
                np.abs(
                    unordered["numeric"].to_numpy()
                    - stored_unordered[
                        "numeric_direction_contrast"
                    ].to_numpy()
                )
            )
        ),
    )

    masks = {
        "all": np.ones(len(unordered), dtype=bool),
        "cross_vocabulary": unordered["cross_vocabulary"].to_numpy(bool),
        "opposite_orientation": unordered[
            "opposite_orientation"
        ].to_numpy(bool),
    }
    masks["both"] = (
        masks["cross_vocabulary"] & masks["opposite_orientation"]
    )
    with np.load(OUT / "primary_exact_donor_label_nulls.npz") as nulls:
        observed_row = int(
            statistics["structured_donor_label_nulls"]["all"][
                "observed_assignment_row"
            ]
        )
        for name, mask in masks.items():
            values = unordered.loc[mask, "physical"].to_numpy()
            observed = float(values.mean())
            stored = statistics["primary"][name]
            structured = statistics["structured_donor_label_nulls"][name]
            check(
                f"{name}: independent endpoint",
                np.isclose(observed, stored["mean"], atol=1e-12),
                independent=observed,
                recorded=stored["mean"],
            )
            sign_p = pair_sign_p(values)
            check(
                f"{name}: independent pair-sign p",
                np.isclose(
                    sign_p,
                    stored["exact_two_sided_p"],
                    atol=1e-15,
                ),
                independent=sign_p,
                recorded=stored["exact_two_sided_p"],
            )
            null = nulls[name]
            structured_p = float(
                np.mean(np.abs(null) >= abs(observed) - 1e-12)
            )
            check(
                f"{name}: structured-null observed identity",
                np.isclose(null[observed_row], observed, atol=1e-12),
                null_observed=float(null[observed_row]),
                endpoint=observed,
            )
            check(
                f"{name}: independent structured p",
                np.isclose(
                    structured_p,
                    structured["exact_two_sided_p"],
                    atol=1e-15,
                ),
                independent=structured_p,
                recorded=structured["exact_two_sided_p"],
            )

    family = (
        ordered.groupby("donor_family", as_index=False)["physical"].mean()
    )
    positive_families = int(np.sum(family["physical"] > 0))
    check(
        "independent donor-family breadth",
        positive_families
        == statistics["frozen_verdict"]["positive_donor_families"]
        == 4,
        independent=positive_families,
        recorded=statistics["frozen_verdict"][
            "positive_donor_families"
        ],
    )
    check(
        "frozen verdict is partial",
        statistics["frozen_verdict"][
            "option_free_cross_mechanism_transfer"
        ]
        == "partial",
        observed=statistics["frozen_verdict"],
    )

    report = {
        "study_id": "cross-mechanism-activation-patching-audit-2026-07-18",
        "passed": all(row["passed"] for row in checks),
        "n_checks": len(checks),
        "n_passed": sum(row["passed"] for row in checks),
        "checks": checks,
        "artifact_sha256": {
            "raw.json": sha256(OUT / "raw.json"),
            "statistics.json": sha256(OUT / "statistics.json"),
            "all_patch_rows.csv": sha256(OUT / "all_patch_rows.csv"),
            "primary_exact_donor_label_nulls.npz": sha256(
                OUT / "primary_exact_donor_label_nulls.npz"
            ),
            "protocol-amendment-v1.json": sha256(
                OUT / "protocol-amendment-v1.json"
            ),
        },
        "guardrail": (
            "This audit recomputes registered summaries independently from "
            "raw patch rows. It adds no inferential endpoint."
        ),
    }
    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not report["passed"]:
        failures = [
            row["name"] for row in checks if not row["passed"]
        ]
        raise SystemExit("audit failed: " + ", ".join(failures))
    print(f"validated {report['n_passed']}/{report['n_checks']} checks")


if __name__ == "__main__":
    main()
