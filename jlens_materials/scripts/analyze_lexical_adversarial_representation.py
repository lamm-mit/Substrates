#!/usr/bin/env python3
"""Analyze the frozen lexical-adversarial materials representation study."""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "lexical-adversarial-representation-2026-07-17"
PROTOCOL_PATH = OUT / "protocol.json"
MANIFEST_PATH = OUT / "prompt_manifest.json"
RAW_PATH = OUT / "raw.json"
STATES_PATH = OUT / "representations.npz"
FIG = OUT / "figures"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def centered_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def uncentered_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def similarity_rows(
    methods: dict[str, np.ndarray],
    layers: np.ndarray,
    triplets: list[dict],
    prompt_index: dict[str, int],
    *,
    centered: bool,
    accumulator_dtype: type[np.floating] = np.float32,
) -> pd.DataFrame:
    output = []
    normalizer = centered_rows if centered else uncentered_rows
    for method, array in methods.items():
        for layer_index, layer in enumerate(layers):
            values = normalizer(array[:, layer_index])
            for triplet in triplets:
                ids = triplet["prompt_ids"]
                anchor = values[prompt_index[ids["anchor"]]]
                paraphrase = values[prompt_index[ids["physics_paraphrase"]]]
                counter = values[prompt_index[ids["lexical_counterfactual"]]]
                # The discovery artifact was frozen with float32 accumulation.
                # Later cohorts used float64 after an implementation-only
                # numerical-stability amendment. The explicit argument lets
                # both historical artifacts remain exactly reproducible.
                ap = float(np.dot(
                    anchor.astype(accumulator_dtype, copy=False),
                    paraphrase.astype(accumulator_dtype, copy=False),
                ))
                ac = float(np.dot(
                    anchor.astype(accumulator_dtype, copy=False),
                    counter.astype(accumulator_dtype, copy=False),
                ))
                output.append({
                    "method": method,
                    "centered": centered,
                    "layer": int(layer),
                    "depth_percent": 100.0 * int(layer) / 41.0,
                    "triplet_id": triplet["triplet_id"],
                    "family_id": triplet["family_id"],
                    "anchor_paraphrase_cosine": ap,
                    "anchor_counterfactual_cosine": ac,
                    "physics_minus_lexical_margin": ap - ac,
                })
    return pd.DataFrame(output)


def two_stage_bootstrap(
    frame: pd.DataFrame,
    value: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    grouped = {
        family: group[value].to_numpy(dtype=float)
        for family, group in frame.groupby("family_id", sort=True)
    }
    families = sorted(grouped)
    family_means = np.asarray([grouped[family].mean() for family in families])
    observed = float(family_means.mean())
    draws = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        selected = rng.integers(0, len(families), size=len(families))
        selected_means = []
        for family_index in selected:
            values = grouped[families[family_index]]
            sample = values[rng.integers(0, len(values), size=len(values))]
            selected_means.append(float(sample.mean()))
        draws[iteration] = float(np.mean(selected_means))
    return {
        "mean": observed,
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_above_zero": float(np.mean(draws > 0)),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def layer_summary_with_ci(
    frame: pd.DataFrame,
    value: str,
    methods: list[str],
    *,
    n_resamples: int = 5000,
) -> pd.DataFrame:
    output = []
    for method in methods:
        for layer, group in frame[frame["method"] == method].groupby("layer"):
            result = two_stage_bootstrap(
                group, value, n_resamples=n_resamples,
                seed=20260717 + int(layer) + 1000 * methods.index(method),
            )
            output.append({
                "method": method,
                "layer": int(layer),
                "depth_percent": float(group["depth_percent"].iloc[0]),
                **result,
            })
    return pd.DataFrame(output)


def exact_family_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product([-1.0, 1.0], repeat=len(values))
    ]
    return float(sum(value >= observed - 1e-15 for value in null) / len(null))


def behavior_summary(raw: dict, manifest: dict) -> tuple[pd.DataFrame, dict]:
    clean = pd.DataFrame(raw["clean_rows"])
    prompt_meta = pd.DataFrame(manifest["prompts"])[[
        "prompt_id", "presentation_order", "outcome_positive", "outcome_negative"
    ]]
    clean = clean.merge(prompt_meta, on="prompt_id", how="left", validate="one_to_one")
    triplet_consistency = []
    for triplet_id, group in clean.groupby("triplet_id"):
        by_variant = group.set_index("variant")
        anchor = by_variant.loc["anchor"]
        paraphrase = by_variant.loc["physics_paraphrase"]
        counter = by_variant.loc["lexical_counterfactual"]
        triplet_consistency.append({
            "triplet_id": triplet_id,
            "family_id": str(anchor["family_id"]),
            "all_three_registered_pair_correct": bool(
                anchor["registered_pair_correct"]
                and paraphrase["registered_pair_correct"]
                and counter["registered_pair_correct"]
            ),
            "anchor_paraphrase_same_prediction": (
                anchor["predicted_registered_outcome"]
                == paraphrase["predicted_registered_outcome"]
            ),
            "counterfactual_reverses_prediction": (
                anchor["predicted_registered_outcome"]
                != counter["predicted_registered_outcome"]
            ),
            "full_scientific_triplet_consistency": bool(
                anchor["registered_pair_correct"]
                and paraphrase["registered_pair_correct"]
                and counter["registered_pair_correct"]
                and anchor["predicted_registered_outcome"]
                == paraphrase["predicted_registered_outcome"]
                and anchor["predicted_registered_outcome"]
                != counter["predicted_registered_outcome"]
            ),
        })
    triplet_frame = pd.DataFrame(triplet_consistency)
    summary = {
        "registered_pair_accuracy": float(clean["registered_pair_correct"].mean()),
        "global_top_registered_answer_rate": float(
            clean["global_top_is_registered_answer"].mean()
        ),
        "by_variant_accuracy": {
            variant: float(group["registered_pair_correct"].mean())
            for variant, group in clean.groupby("variant")
        },
        "triplets_all_three_correct": int(
            triplet_frame["all_three_registered_pair_correct"].sum()
        ),
        "triplets_full_scientific_consistency": int(
            triplet_frame["full_scientific_triplet_consistency"].sum()
        ),
        "n_triplets": len(triplet_frame),
        "family_accuracy": {
            family: float(group["registered_pair_correct"].mean())
            for family, group in clean.groupby("family_id")
        },
    }
    return clean, {**summary, "triplet_rows": triplet_consistency}


def prompt_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z'-]{2,}", text.lower()))


def keep_candidate(token: str, removed_words: set[str], stopwords: set[str]) -> bool:
    token = token.lower().strip()
    if not re.fullmatch(r"[a-z][a-z-]{2,}", token):
        return False
    if token in stopwords or token in removed_words:
        return False
    if any(
        min(len(token), len(word)) >= 5
        and (token.startswith(word) or word.startswith(token))
        for word in removed_words
    ):
        return False
    return True


def target_free_analysis(
    raw: dict,
    manifest: dict,
    protocol: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompts = {row["prompt_id"]: row for row in manifest["prompts"]}
    rows = {
        (row["prompt_id"], row["method"], int(row["layer"])): row
        for row in raw["target_free_top_tokens"]
    }
    output = []
    examples = []
    display_k = int(protocol["target_free_display_k"])
    stopwords = set(protocol["frozen_stopwords"])
    for triplet_index, triplet in enumerate(manifest["triplets"]):
        ids = triplet["prompt_ids"]
        removed = set()
        for prompt_id in ids.values():
            removed |= prompt_words(prompts[prompt_id]["user"])
        for method in ["direct", "jacobian_ensemble"]:
            for layer in protocol["target_free_layers"]:
                candidates = {}
                for variant, prompt_id in ids.items():
                    source = rows[(prompt_id, method, int(layer))]
                    filtered = []
                    seen = set()
                    for token in source["tokens"]:
                        canonical = token.lower().strip()
                        if canonical in seen:
                            continue
                        if keep_candidate(canonical, removed, stopwords):
                            seen.add(canonical)
                            filtered.append(canonical)
                        if len(filtered) == display_k:
                            break
                    candidates[variant] = filtered
                anchor = set(candidates["anchor"])
                paraphrase = set(candidates["physics_paraphrase"])
                counter = set(candidates["lexical_counterfactual"])
                ap_union = anchor | paraphrase
                ac_union = anchor | counter
                ap = len(anchor & paraphrase) / len(ap_union) if ap_union else 0.0
                ac = len(anchor & counter) / len(ac_union) if ac_union else 0.0
                output.append({
                    "triplet_id": triplet["triplet_id"],
                    "family_id": triplet["family_id"],
                    "method": method,
                    "layer": int(layer),
                    "depth_percent": 100.0 * int(layer) / 41.0,
                    "anchor_paraphrase_jaccard": ap,
                    "anchor_counterfactual_jaccard": ac,
                    "physics_minus_lexical_jaccard": ap - ac,
                    "anchor_n": len(anchor),
                    "paraphrase_n": len(paraphrase),
                    "counterfactual_n": len(counter),
                })
                if triplet_index % 4 == 0 and int(layer) == 28:
                    for variant in [
                        "anchor", "physics_paraphrase", "lexical_counterfactual"
                    ]:
                        examples.append({
                            "triplet_id": triplet["triplet_id"],
                            "family_id": triplet["family_id"],
                            "method": method,
                            "layer": int(layer),
                            "variant": variant,
                            "candidates": " | ".join(candidates[variant]),
                        })
    return pd.DataFrame(output), pd.DataFrame(examples)


def make_figure(
    layer_frame: pd.DataFrame,
    band_frame: pd.DataFrame,
    family_frame: pd.DataFrame,
    lexical_rows: pd.DataFrame,
    target_free_frame: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })
    colors = {
        "jacobian_ensemble": "#167D8D",
        "direct": "#6F5AA8",
    }
    labels = {
        "jacobian_ensemble": "Jacobian",
        "direct": "direct",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    layer_summary = layer_summary_with_ci(
        layer_frame,
        "physics_minus_lexical_margin",
        ["jacobian_ensemble", "direct"],
    )
    for method in ["jacobian_ensemble", "direct"]:
        subset = layer_summary[layer_summary["method"] == method].sort_values("layer")
        x = subset["depth_percent"].to_numpy()
        mean = subset["mean"].to_numpy()
        low = np.asarray([value[0] for value in subset["ci95"]])
        high = np.asarray([value[1] for value in subset["ci95"]])
        ax_a.plot(x, mean, color=colors[method], linewidth=1.7, label=labels[method])
        ax_a.fill_between(x, low, high, color=colors[method], alpha=0.15, linewidth=0)
    ax_a.axhline(0, color="#666666", linewidth=0.8)
    ax_a.axvspan(38, 92, color="#999999", alpha=0.08, linewidth=0)
    ax_a.set_xlabel("Layer depth (%)")
    ax_a.set_ylabel("Physics-equivalence margin\n(centered cosine units)")
    ax_a.text(0.02, 0.97, "A", transform=ax_a.transAxes, va="top", fontweight="bold")

    family_order = list(dict.fromkeys(family_frame["family_id"]))
    x = np.arange(len(family_order))
    offsets = {"jacobian_ensemble": -0.14, "direct": 0.14}
    for method in ["jacobian_ensemble", "direct"]:
        subset = (
            family_frame[family_frame["method"] == method]
            .set_index("family_id")
            .loc[family_order]
        )
        ax_b.scatter(
            x + offsets[method],
            subset["physics_minus_lexical_margin"],
            s=38,
            color=colors[method],
            marker="o" if method == "jacobian_ensemble" else "^",
            label=labels[method],
        )
    ax_b.axhline(0, color="#666666", linewidth=0.8)
    short_family = {
        "grain-size-strength": "grain\nsize",
        "crack-size-fracture": "crack\nsize",
        "temperature-diffusion": "temperature–\ndiffusion",
        "fiber-angle-stiffness": "fiber\nangle",
        "fiber-fraction-stiffness": "fiber\nfraction",
        "martensite-hardness": "martensite\nfraction",
    }
    ax_b.set_xticks(x, [short_family[value] for value in family_order])
    ax_b.set_ylabel("Band-mean physics-equivalence margin")
    ax_b.text(0.02, 0.97, "B", transform=ax_b.transAxes, va="top", fontweight="bold")

    jacobian_band = band_frame[
        band_frame["method"] == "jacobian_ensemble"
    ].set_index("triplet_id")
    lexical = lexical_rows.set_index("triplet_id")
    family_codes = {family: index for index, family in enumerate(family_order)}
    scatter = ax_c.scatter(
        lexical.loc[jacobian_band.index, "word_tfidf_margin"],
        jacobian_band["physics_minus_lexical_margin"],
        c=[family_codes[value] for value in jacobian_band["family_id"]],
        cmap="tab10",
        s=34,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_c.axhline(0, color="#666666", linewidth=0.8)
    ax_c.axvline(0, color="#666666", linewidth=0.8)
    ax_c.set_xlabel("Word TF–IDF margin")
    ax_c.set_ylabel("Jacobian band margin")
    ax_c.text(0.02, 0.97, "C", transform=ax_c.transAxes, va="top", fontweight="bold")

    target_summary = layer_summary_with_ci(
        target_free_frame.rename(
            columns={"physics_minus_lexical_jaccard": "margin"}
        ),
        "margin",
        ["jacobian_ensemble", "direct"],
    )
    for method in ["jacobian_ensemble", "direct"]:
        subset = target_summary[
            target_summary["method"] == method
        ].sort_values("layer")
        x_values = subset["depth_percent"].to_numpy()
        means = subset["mean"].to_numpy()
        low = np.asarray([value[0] for value in subset["ci95"]])
        high = np.asarray([value[1] for value in subset["ci95"]])
        ax_d.errorbar(
            x_values,
            means,
            yerr=[means - low, high - means],
            color=colors[method],
            marker="o" if method == "jacobian_ensemble" else "^",
            linewidth=1.4,
            capsize=2,
            label=labels[method],
        )
    ax_d.axhline(0, color="#666666", linewidth=0.8)
    ax_d.axvspan(38, 92, color="#999999", alpha=0.08, linewidth=0)
    ax_d.set_xlabel("Layer depth (%)")
    ax_d.set_ylabel("Target-free word-set margin\n(Jaccard units)")
    ax_d.text(0.02, 0.97, "D", transform=ax_d.transAxes, va="top", fontweight="bold")

    handles, legend_labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(
            FIG / f"lexical-adversarial-representation.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if raw["provenance"]["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("raw output does not match the frozen protocol")
    if protocol["inputs"]["prompt_manifest_sha256"] != sha256(MANIFEST_PATH):
        raise RuntimeError("prompt manifest fingerprint mismatch")
    arrays = np.load(STATES_PATH)
    prompt_ids = [str(value) for value in arrays["prompt_ids"]]
    layers = arrays["layers"].astype(int)
    expected_prompt_ids = [row["prompt_id"] for row in manifest["prompts"]]
    if prompt_ids != expected_prompt_ids:
        raise RuntimeError("state prompt order does not match manifest")
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    jacobian = arrays["jacobian_decoder_basis"].astype(np.float32)
    methods = {
        "raw_residual": arrays["raw_states"].astype(np.float32),
        "direct": arrays["direct_decoder_basis"].astype(np.float32),
        "jacobian_seed0": jacobian[0],
        "jacobian_seed1": jacobian[1],
        "jacobian_seed2": jacobian[2],
        "jacobian_ensemble": jacobian.mean(axis=0),
    }
    centered = similarity_rows(
        methods, layers, manifest["triplets"], prompt_index, centered=True
    )
    uncentered = similarity_rows(
        methods, layers, manifest["triplets"], prompt_index, centered=False
    )
    layer_frame = pd.concat([centered, uncentered], ignore_index=True)
    layer_frame.to_csv(OUT / "layer_similarity_margins.csv", index=False)

    band = centered[
        centered["depth_percent"].between(
            protocol["registered_band_percent"][0],
            protocol["registered_band_percent"][1],
        )
    ]
    band_frame = (
        band.groupby(["method", "triplet_id", "family_id"], as_index=False)[
            "physics_minus_lexical_margin"
        ]
        .mean()
    )
    band_frame.to_csv(OUT / "band_triplet_margins.csv", index=False)
    family_frame = (
        band_frame.groupby(["method", "family_id"], as_index=False)[
            "physics_minus_lexical_margin"
        ]
        .mean()
    )
    family_frame.to_csv(OUT / "family_band_margins.csv", index=False)

    bootstrap = {
        method: two_stage_bootstrap(
            band_frame[band_frame["method"] == method],
            "physics_minus_lexical_margin",
            n_resamples=30000,
            seed=20260717 + method_index,
        )
        for method_index, method in enumerate(methods)
    }
    pivot = band_frame.pivot(
        index=["triplet_id", "family_id"],
        columns="method",
        values="physics_minus_lexical_margin",
    ).reset_index()
    pivot["jacobian_minus_direct"] = (
        pivot["jacobian_ensemble"] - pivot["direct"]
    )
    contrast = two_stage_bootstrap(
        pivot,
        "jacobian_minus_direct",
        n_resamples=30000,
        seed=20260817,
    )
    contrast_family = (
        pivot.groupby("family_id")["jacobian_minus_direct"].mean().to_numpy()
    )
    contrast["exact_family_sign_flip_p"] = exact_family_sign_flip(contrast_family)

    lexical_protocol = protocol["lexical_adversarial_preflight"]
    lexical_by_triplet = pd.DataFrame({
        "triplet_id": [
            row["triplet_id"] for row in lexical_protocol["word_tfidf"]["rows"]
        ],
        "word_tfidf_margin": [
            row["physics_minus_lexical_margin"]
            for row in lexical_protocol["word_tfidf"]["rows"]
        ],
        "character_tfidf_margin": [
            row["physics_minus_lexical_margin"]
            for row in lexical_protocol["character_tfidf"]["rows"]
        ],
    })
    lexical_by_triplet.to_csv(OUT / "lexical_baseline_margins.csv", index=False)

    clean_frame, behavior = behavior_summary(raw, manifest)
    clean_frame.to_csv(OUT / "clean_behavior.csv", index=False)
    pd.DataFrame(behavior["triplet_rows"]).to_csv(
        OUT / "clean_triplet_consistency.csv", index=False
    )

    target_free, examples = target_free_analysis(raw, manifest, protocol)
    target_free.to_csv(OUT / "target_free_jaccard_margins.csv", index=False)
    examples.to_csv(OUT / "target_free_candidate_examples.csv", index=False)
    selected_band_layers = [
        int(layer) for layer in protocol["target_free_layers"]
        if protocol["registered_band_percent"][0]
        <= 100.0 * int(layer) / 41.0
        <= protocol["registered_band_percent"][1]
    ]
    target_band = (
        target_free[target_free["layer"].isin(selected_band_layers)]
        .groupby(["method", "triplet_id", "family_id"], as_index=False)[
            "physics_minus_lexical_jaccard"
        ]
        .mean()
    )
    target_stats = {
        method: two_stage_bootstrap(
            target_band[target_band["method"] == method],
            "physics_minus_lexical_jaccard",
            n_resamples=30000,
            seed=20260917 + method_index,
        )
        for method_index, method in enumerate(["jacobian_ensemble", "direct"])
    }

    primary_rows = band_frame[
        band_frame["method"] == "jacobian_ensemble"
    ]
    primary_family = family_frame[
        family_frame["method"] == "jacobian_ensemble"
    ]
    primary_positive_triplets = int(
        (primary_rows["physics_minus_lexical_margin"] > 0).sum()
    )
    primary_positive_families = int(
        (primary_family["physics_minus_lexical_margin"] > 0).sum()
    )
    decision = {
        "primary_ci_above_zero": bootstrap["jacobian_ensemble"]["ci95"][0] > 0,
        "positive_triplets": primary_positive_triplets,
        "triplet_breadth_pass": primary_positive_triplets >= 18,
        "positive_families": primary_positive_families,
        "family_breadth_pass": primary_positive_families >= 5,
    }
    decision["frozen_success_rule_pass"] = all([
        decision["primary_ci_above_zero"],
        decision["triplet_breadth_pass"],
        decision["family_breadth_pass"],
    ])

    statistics = {
        "study_id": protocol["study_id"],
        "status": protocol["status"],
        "dimensions": raw["dimensions"],
        "provenance": {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "manifest_sha256": sha256(MANIFEST_PATH),
            "raw_sha256": sha256(RAW_PATH),
            "representations_sha256": sha256(STATES_PATH),
        },
        "lexical_preflight": {
            key: {
                field: value for field, value in result.items() if field != "rows"
            }
            for key, result in lexical_protocol.items()
        },
        "behavior": {
            key: value for key, value in behavior.items() if key != "triplet_rows"
        },
        "centered_band_bootstrap": bootstrap,
        "jacobian_minus_direct": contrast,
        "primary_breadth": {
            "positive_triplets": primary_positive_triplets,
            "n_triplets": len(primary_rows),
            "exact_two_sided_sign_p": float(stats.binomtest(
                primary_positive_triplets,
                len(primary_rows),
                p=0.5,
                alternative="two-sided",
            ).pvalue),
            "positive_families": primary_positive_families,
            "n_families": len(primary_family),
        },
        "family_band_margins": {
            method: {
                row["family_id"]: float(row["physics_minus_lexical_margin"])
                for _, row in family_frame[family_frame["method"] == method].iterrows()
            }
            for method in methods
        },
        "target_free_selected_layers": selected_band_layers,
        "target_free_jaccard_bootstrap": target_stats,
        "decision": decision,
        "guardrails": protocol["guardrails"],
    }
    dump_json(OUT / "statistics.json", statistics)
    make_figure(
        centered[centered["method"].isin(["jacobian_ensemble", "direct"])],
        band_frame,
        family_frame,
        lexical_by_triplet,
        target_free,
    )

    primary = bootstrap["jacobian_ensemble"]
    direct = bootstrap["direct"]
    raw_result = bootstrap["raw_residual"]
    target_j = target_stats["jacobian_ensemble"]
    target_d = target_stats["direct"]
    lines = [
        "# Prospective lexical-adversarial representation results",
        "",
        "## Frozen question",
        "",
        (
            "Does Gemma represent a differently worded but physically equivalent "
            "materials description as closer than a near-verbatim prompt in "
            "which only the physical relation is reversed?"
        ),
        "",
        "## Design check",
        "",
        (
            f"Word and character TF-IDF selected the lexical counterfactual in "
            f"all 24 triplets. Their mean physics-minus-lexical margins were "
            f"{lexical_protocol['word_tfidf']['mean_physics_minus_lexical_margin']:+.3f} "
            f"and {lexical_protocol['character_tfidf']['mean_physics_minus_lexical_margin']:+.3f}. "
            "Negative is the intended lexical trap."
        ),
        "",
        "## Primary result",
        "",
        (
            f"The frozen centered Jacobian ensemble band margin was "
            f"**{primary['mean']:+.4f}** (two-stage 95% CI "
            f"{primary['ci95'][0]:+.4f} to {primary['ci95'][1]:+.4f}). "
            f"{primary_positive_triplets}/24 triplets and "
            f"{primary_positive_families}/6 family means were positive. "
            f"Frozen decision: **{'PASS' if decision['frozen_success_rule_pass'] else 'FAIL'}**."
        ),
        "",
        (
            f"The matched direct-decoder-basis result was {direct['mean']:+.4f} "
            f"({direct['ci95'][0]:+.4f} to {direct['ci95'][1]:+.4f}); raw "
            f"residual states gave {raw_result['mean']:+.4f} "
            f"({raw_result['ci95'][0]:+.4f} to {raw_result['ci95'][1]:+.4f}). "
            f"The paired Jacobian-minus-direct contrast was "
            f"{contrast['mean']:+.4f} ({contrast['ci95'][0]:+.4f} to "
            f"{contrast['ci95'][1]:+.4f}; exact family sign-flip "
            f"p={contrast['exact_family_sign_flip_p']:.4f})."
        ),
        "",
        "## Behavioral audit",
        "",
        (
            f"Registered-pair answer accuracy was "
            f"{behavior['registered_pair_accuracy']:.1%}; the registered answer "
            f"was the global top token in "
            f"{behavior['global_top_registered_answer_rate']:.1%} of prompts. "
            f"{behavior['triplets_full_scientific_consistency']}/24 triplets "
            "gave the same correct decision for anchor and paraphrase and the "
            "opposite correct decision for the lexical counterfactual."
        ),
        "",
        "## Target-free vocabulary",
        "",
        (
            f"Across frozen in-band decoding layers, the target-free top-word "
            f"Jaccard margin was {target_j['mean']:+.4f} "
            f"({target_j['ci95'][0]:+.4f} to {target_j['ci95'][1]:+.4f}) for "
            f"the Jacobian ensemble and {target_d['mean']:+.4f} "
            f"({target_d['ci95'][0]:+.4f} to {target_d['ci95'][1]:+.4f}) for "
            "direct decoding. Positive means that emergent word neighborhoods "
            "follow physical equivalence more than near-verbatim wording."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "A positive similarity margin shows representational organization "
            "under a deliberately adversarial lexical test. It is not a causal "
            "intervention, a literal hidden chain of thought, or evidence that "
            "the model can solve arbitrary materials problems. All prompts, "
            "families, layers, clean errors, and target-free candidates are retained."
        ),
        "",
        "## Files",
        "",
        "- `prompt_manifest.json`: all 72 exact prompts.",
        "- `protocol.json` and `PROTOCOL.md`: frozen endpoints and fingerprints.",
        "- `raw.json`: clean answers and 720 target-free top-token records.",
        "- `representations.npz`: raw, direct-basis, and three-fit Jacobian states.",
        "- `layer_similarity_margins.csv`: every triplet-by-layer similarity.",
        "- `band_triplet_margins.csv`: primary independent-unit table.",
        "- `family_band_margins.csv`: family-level summary.",
        "- `target_free_jaccard_margins.csv`: target-free neighborhood comparison.",
        "- `target_free_candidate_examples.csv`: fixed first-case examples.",
        "- `statistics.json`: complete numerical decision record.",
        "- `figures/lexical-adversarial-representation.{png,pdf}`: candidate figure.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    (OUT / "README.md").write_text(
        "\n".join([
            "# Lexical-adversarial materials representation study",
            "",
            (
                "Frozen discovery cohort reported in the paper and "
                "Supplementary Information."
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/run_lexical_adversarial_representation.py --device mps --dtype bfloat16",
            "python scripts/analyze_lexical_adversarial_representation.py",
            "```",
            "",
            "Read `PROTOCOL.md` before `RESULTS.md`. The protocol and exact prompt "
            "manifest were checksum-locked before any model forward pass.",
            "",
        ])
    )
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
