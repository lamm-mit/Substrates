#!/usr/bin/env python3
"""Analyze the frozen fresh-law physical-abstraction confirmation."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "experiments" / "elicited-physics-abstraction-2026-07-18"
OUT = ROOT / "experiments" / "fresh-law-abstraction-confirmation-2026-07-18"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(scale, 1e-12)


def fit_centroid(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = values[labels].mean(axis=0)
    negative = values[~labels].mean(axis=0)
    direction = positive - negative
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return direction, 0.5 * (positive + negative)


def project(values: np.ndarray, direction: np.ndarray, midpoint: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ij,j->i", values - midpoint, direction, optimize=False
    )


def load(path: Path) -> dict:
    manifest = json.loads((path / "prompt_manifest.json").read_text())
    raw = json.loads((path / "raw.json").read_text())
    with np.load(path / "representations.npz", allow_pickle=False) as arrays:
        prompt_ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        states = arrays["raw_states"].astype(np.float64)
    by_id = {row["prompt_id"]: row for row in manifest["prompts"]}
    prompts = [by_id[x] for x in prompt_ids]
    clean_by_id = {row["prompt_id"]: row for row in raw["clean_rows"]}
    clean = [clean_by_id[x] for x in prompt_ids]
    return {
        "manifest": manifest,
        "raw": raw,
        "prompts": prompts,
        "clean": clean,
        "positions": positions,
        "layers": layers,
        "states": states,
    }


def meta(data: dict) -> dict[str, np.ndarray]:
    prompts = data["prompts"]
    return {
        "law": np.asarray([x["law_id"] for x in prompts]),
        "law_sign": np.asarray([x["law_sign"] > 0 for x in prompts]),
        "numeric": np.asarray([x["numerical_sign"] > 0 for x in prompts]),
        "physical": np.asarray([x["physical_sign"] > 0 for x in prompts]),
        "surface": np.asarray([x["surface"] for x in prompts]),
        "order": np.asarray([x.get("answer_order", "fixed") for x in prompts]),
        "correct": np.asarray([bool(x["correct"]) for x in data["clean"]]),
    }


def per_law_auc(
    scores: np.ndarray, labels: np.ndarray, law: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    if mask is None:
        mask = np.ones(len(scores), dtype=bool)
    output = {}
    for law_id in sorted(set(law[mask])):
        selected = mask & (law == law_id)
        output[str(law_id)] = float(
            roc_auc_score(labels[selected].astype(int), scores[selected])
        )
    return output


def bootstrap(values: np.ndarray, seed: int, n: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for start in range(0, n, 5000):
        count = min(5000, n - start)
        sample = rng.choice(values, size=(count, len(values)), replace=True)
        means[start : start + count] = sample.mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def exact_signflip(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - 0.5
    observed = abs(float(centered.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(centered)):
        statistic = abs(float(np.mean(centered * np.asarray(signs))))
        count += statistic >= observed - 1e-15
        total += 1
    return float(count / total)


def tfidf_control(dev: dict, fresh: dict) -> list[dict]:
    dm, fm = meta(dev), meta(fresh)
    dev_text = [x["user"] for x in dev["prompts"]]
    fresh_text = [x["user"] for x in fresh["prompts"]]
    rows = []
    vectorizers = {
        "word_tfidf": TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        ),
        "character_tfidf": TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30000,
            sublinear_tf=True,
        ),
    }
    for name, vectorizer in vectorizers.items():
        dev_values = normalize(vectorizer.fit_transform(dev_text).toarray())
        fresh_values = normalize(vectorizer.transform(fresh_text).toarray())
        direction, midpoint = fit_centroid(dev_values, dm["physical"])
        scores = project(fresh_values, direction, midpoint)
        law_auc = per_law_auc(scores, fm["physical"], fm["law"])
        for law_id, auc in law_auc.items():
            rows.append({"method": name, "law_id": law_id, "auc": auc})
    return rows


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    dev = load(DEV)
    fresh = load(OUT)
    for name, record in protocol["inputs"].items():
        if name in {"prompt_manifest", "runner"}:
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    dm, fm = meta(dev), meta(fresh)
    layer = int(protocol["primary_decoder"]["layer"])
    position = protocol["primary_decoder"]["position"]
    dev_pi = int(np.flatnonzero(dev["positions"] == position)[0])
    fresh_pi = int(np.flatnonzero(fresh["positions"] == position)[0])
    dev_li = int(np.flatnonzero(dev["layers"] == layer)[0])
    fresh_li = int(np.flatnonzero(fresh["layers"] == layer)[0])
    dev_values = normalize(dev["states"][dev_pi, :, dev_li])
    fresh_values = normalize(fresh["states"][fresh_pi, :, fresh_li])
    direction, midpoint = fit_centroid(dev_values, dm["physical"])
    scores = project(fresh_values, direction, midpoint)

    law_auc = per_law_auc(scores, fm["physical"], fm["law"])
    numeric_auc = per_law_auc(scores, fm["numeric"], fm["law"])
    records = []
    for law_id, auc in law_auc.items():
        selected = fm["law"] == law_id
        records.append(
            {
                "law_id": law_id,
                "law_orientation": (
                    "direct" if bool(fm["law_sign"][selected][0]) else "inverse"
                ),
                "physical_auc": auc,
                "numeric_auc": numeric_auc[law_id],
                "behavior_accuracy": float(fm["correct"][selected].mean()),
                "surface_a_auc": per_law_auc(
                    scores, fm["physical"], fm["law"], fm["surface"] == "a"
                )[law_id],
                "surface_b_auc": per_law_auc(
                    scores, fm["physical"], fm["law"], fm["surface"] == "b"
                )[law_id],
                "higher_first_auc": per_law_auc(
                    scores,
                    fm["physical"],
                    fm["law"],
                    fm["order"] == "higher-first",
                )[law_id],
                "lower_first_auc": per_law_auc(
                    scores,
                    fm["physical"],
                    fm["law"],
                    fm["order"] == "lower-first",
                )[law_id],
            }
        )
    table = pd.DataFrame(records)
    table.to_csv(OUT / "fresh_law_primary.csv", index=False)

    # Descriptive layer profile. The decoder at every layer is still fit only
    # on old laws; the frozen primary remains layer 34.
    layer_rows = []
    for layer_value in dev["layers"]:
        dli = int(np.flatnonzero(dev["layers"] == layer_value)[0])
        fli = int(np.flatnonzero(fresh["layers"] == layer_value)[0])
        dv = normalize(dev["states"][dev_pi, :, dli])
        fv = normalize(fresh["states"][fresh_pi, :, fli])
        d, mid = fit_centroid(dv, dm["physical"])
        s = project(fv, d, mid)
        values = np.asarray(
            list(per_law_auc(s, fm["physical"], fm["law"]).values())
        )
        layer_rows.append(
            {
                "layer": int(layer_value),
                "mean_fresh_law_auc": float(values.mean()),
                "positive_laws": int(np.sum(values > 0.5)),
            }
        )
    pd.DataFrame(layer_rows).to_csv(OUT / "descriptive_layer_profile.csv", index=False)

    tfidf = pd.DataFrame(tfidf_control(dev, fresh))
    tfidf.to_csv(OUT / "tfidf_controls.csv", index=False)
    primary_values = table["physical_auc"].to_numpy()
    success = protocol["success_rule"]
    surface_means = {
        "a": float(table["surface_a_auc"].mean()),
        "b": float(table["surface_b_auc"].mean()),
    }
    order_means = {
        "higher_first": float(table["higher_first_auc"].mean()),
        "lower_first": float(table["lower_first_auc"].mean()),
    }
    summary = {
        "study_id": protocol["study_id"],
        "analysis_sha256": sha256(Path(__file__).resolve()),
        "primary": {
            "position": position,
            "layer": layer,
            "mean_fresh_law_auc": float(primary_values.mean()),
            "bootstrap_95": bootstrap(
                primary_values,
                protocol["inference"]["bootstrap_seed"],
                protocol["inference"]["bootstrap_resamples"],
            ),
            "exact_signflip_p": exact_signflip(primary_values),
            "positive_laws": int(np.sum(primary_values > 0.5)),
            "laws_total": int(len(primary_values)),
            "direct_mean_auc": float(
                table.loc[table["law_orientation"] == "direct", "physical_auc"].mean()
            ),
            "inverse_mean_auc": float(
                table.loc[table["law_orientation"] == "inverse", "physical_auc"].mean()
            ),
            "surface_mean_auc": surface_means,
            "answer_order_mean_auc": order_means,
            "behavior_accuracy": float(fm["correct"].mean()),
            "numeric_auc_direct": float(
                table.loc[table["law_orientation"] == "direct", "numeric_auc"].mean()
            ),
            "numeric_auc_inverse": float(
                table.loc[table["law_orientation"] == "inverse", "numeric_auc"].mean()
            ),
        },
        "controls": {
            name: {
                "mean_fresh_law_auc": float(group["auc"].mean()),
                "positive_laws": int(np.sum(group["auc"] > 0.5)),
            }
            for name, group in tfidf.groupby("method")
        },
    }
    p = summary["primary"]
    summary["passed_all_frozen_criteria"] = bool(
        p["mean_fresh_law_auc"] >= success["mean_fresh_law_auc_minimum"]
        and p["bootstrap_95"][0] > success["bootstrap_95_lower_above"]
        and p["positive_laws"] >= success["positive_laws_minimum"]
        and p["behavior_accuracy"] >= success["behavior_accuracy_minimum"]
        and min(surface_means.values())
        >= success["both_surface_mean_auc_minimum"]
        and min(order_means.values())
        >= success["both_answer_order_mean_auc_minimum"]
    )
    (OUT / "statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    # Compact audit figure.
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    profile = pd.DataFrame(layer_rows)
    axes[0, 0].plot(profile["layer"], profile["mean_fresh_law_auc"], color="#2468a2", lw=2)
    axes[0, 0].axhline(0.5, color="0.55", ls="--", lw=1)
    axes[0, 0].axvline(layer, color="#b23a48", ls=":", lw=1.5)
    axes[0, 0].scatter([layer], [p["mean_fresh_law_auc"]], color="#b23a48", zorder=3)
    axes[0, 0].set(xlabel="Layer", ylabel="Mean AUC across 16 fresh laws")
    axes[0, 0].text(0.02, 0.96, "A", transform=axes[0, 0].transAxes, va="top", fontweight="bold")

    ordered = table.sort_values("physical_auc")
    colors = ordered["law_orientation"].map({"direct": "#2a9d8f", "inverse": "#7b5ea7"})
    axes[0, 1].barh(range(len(ordered)), ordered["physical_auc"], color=colors)
    axes[0, 1].axvline(0.5, color="0.35", ls="--", lw=1)
    axes[0, 1].set_yticks(range(len(ordered)), ordered["law_id"].str.replace("-", " "), fontsize=6.5)
    axes[0, 1].set(xlabel="Frozen physical-direction AUC", xlim=(0, 1.02))
    axes[0, 1].text(0.02, 0.96, "B", transform=axes[0, 1].transAxes, va="top", fontweight="bold")

    x = np.arange(2)
    physical = [
        p["direct_mean_auc"],
        p["inverse_mean_auc"],
    ]
    numeric = [p["numeric_auc_direct"], p["numeric_auc_inverse"]]
    axes[1, 0].bar(x - 0.18, physical, 0.36, label="Physical outcome", color="#2468a2")
    axes[1, 0].bar(x + 0.18, numeric, 0.36, label="Numerical direction", color="#d79a32")
    axes[1, 0].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[1, 0].set_xticks(x, ["Direct laws", "Inverse laws"])
    axes[1, 0].set(ylabel="Mean within-law AUC", ylim=(0, 1.03))
    axes[1, 0].legend(frameon=False, loc="upper center", ncol=2, fontsize=7)
    axes[1, 0].text(0.02, 0.96, "C", transform=axes[1, 0].transAxes, va="top", fontweight="bold")

    labels = ["surface A", "surface B", "higher first", "lower first", "word TF-IDF", "char TF-IDF"]
    values = [
        surface_means["a"],
        surface_means["b"],
        order_means["higher_first"],
        order_means["lower_first"],
        summary["controls"]["word_tfidf"]["mean_fresh_law_auc"],
        summary["controls"]["character_tfidf"]["mean_fresh_law_auc"],
    ]
    axes[1, 1].bar(range(len(values)), values, color=["#5b8db8"] * 4 + ["0.65", "0.45"])
    axes[1, 1].axhline(0.5, color="0.45", ls="--", lw=1)
    axes[1, 1].set_xticks(range(len(values)), labels, rotation=28, ha="right")
    axes[1, 1].set(ylabel="Mean fresh-law AUC", ylim=(0, 1.03))
    axes[1, 1].text(0.02, 0.96, "D", transform=axes[1, 1].transAxes, va="top", fontweight="bold")
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "fresh-law-abstraction.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "figures" / "fresh-law-abstraction.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Fresh-law abstraction confirmation",
        "",
        f"Frozen layer/position: layer {layer}, `{position}`.",
        f"Behavioral accuracy: {p['behavior_accuracy']:.3f}.",
        (
            f"Primary mean law AUC: {p['mean_fresh_law_auc']:.3f} "
            f"(law bootstrap 95% CI {p['bootstrap_95'][0]:.3f}–"
            f"{p['bootstrap_95'][1]:.3f}; exact p={p['exact_signflip_p']:.5g}; "
            f"{p['positive_laws']}/{p['laws_total']} laws above 0.5)."
        ),
        (
            f"Direct/inverse means: {p['direct_mean_auc']:.3f}/"
            f"{p['inverse_mean_auc']:.3f}."
        ),
        f"Surface means: {surface_means}.",
        f"Answer-order means: {order_means}.",
        f"Lexical controls: {summary['controls']}.",
        f"Passed every frozen criterion: {summary['passed_all_frozen_criteria']}.",
        "",
        "The physical label is the product of law sign and numerical-change sign. "
        "Consequently, a pure numerical-comparison direction must reverse its "
        "apparent meaning between direct and inverse laws; panel C audits this.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
