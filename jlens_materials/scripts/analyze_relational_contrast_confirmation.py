#!/usr/bin/env python3
"""Analyze the frozen second-fresh relational physical contrast."""

from __future__ import annotations

import hashlib
import itertools
import json
from math import comb
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
OUT = ROOT / "experiments" / "relational-contrast-confirmation-2026-07-18"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_centroid(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, neg = x[y].mean(0), x[~y].mean(0)
    direction = pos - neg
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return direction, 0.5 * (pos + neg)


def project(x: np.ndarray, d: np.ndarray, midpoint: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", x - midpoint, d, optimize=False)


def load(path: Path) -> dict:
    manifest = json.loads((path / "prompt_manifest.json").read_text())
    raw = json.loads((path / "raw.json").read_text())
    with np.load(path / "representations.npz", allow_pickle=False) as arrays:
        ids = arrays["prompt_ids"].astype(str)
        positions = arrays["positions"].astype(str)
        layers = arrays["layers"].astype(int)
        states = arrays["raw_states"].astype(np.float64)
    by_id = {x["prompt_id"]: x for x in manifest["prompts"]}
    clean_by_id = {x["prompt_id"]: x for x in raw["clean_rows"]}
    return {
        "manifest": manifest,
        "raw": raw,
        "prompts": [by_id[x] for x in ids],
        "clean": [clean_by_id[x] for x in ids],
        "positions": positions,
        "layers": layers,
        "states": states,
    }


def arrays(data: dict) -> dict[str, np.ndarray]:
    p = data["prompts"]
    return {
        "law": np.asarray([x["law_id"] for x in p]),
        "law_sign": np.asarray([x["law_sign"] > 0 for x in p]),
        "physical": np.asarray([x["physical_sign"] > 0 for x in p]),
        "numeric": np.asarray([x["numerical_sign"] > 0 for x in p]),
        "surface": np.asarray([x["surface"] for x in p]),
        "case": np.asarray([x["case_index"] for x in p]),
        "order": np.asarray([x.get("answer_order", "fixed") for x in p]),
    }


def contrasts(scores: np.ndarray, meta: dict, surface: str | None = None) -> pd.DataFrame:
    rows = []
    for law_id in sorted(set(meta["law"])):
        for formula_surface in sorted(set(meta["surface"])):
            if surface is not None and formula_surface != surface:
                continue
            for case in sorted(set(meta["case"])):
                for order in sorted(set(meta["order"])):
                    base = (
                        (meta["law"] == law_id)
                        & (meta["surface"] == formula_surface)
                        & (meta["case"] == case)
                        & (meta["order"] == order)
                    )
                    up = base & meta["numeric"]
                    down = base & ~meta["numeric"]
                    if up.sum() != 1 or down.sum() != 1:
                        continue
                    rows.append(
                        {
                            "law_id": str(law_id),
                            "law_orientation": (
                                "direct" if bool(meta["law_sign"][up][0]) else "inverse"
                            ),
                            "surface": str(formula_surface),
                            "case_index": int(case),
                            "answer_order": str(order),
                            "contrast": float(scores[up][0] - scores[down][0]),
                        }
                    )
    return pd.DataFrame(rows)


def law_table(pair_table: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_table.groupby(["law_id", "law_orientation"], as_index=False)
        .agg(
            mean_contrast=("contrast", "mean"),
            surface_a_contrast=("contrast", lambda x: float(x[pair_table.loc[x.index, "surface"] == "a"].mean())),
            surface_b_contrast=("contrast", lambda x: float(x[pair_table.loc[x.index, "surface"] == "b"].mean())),
            pair_sd=("contrast", "std"),
            n_pairs=("contrast", "size"),
        )
    )


def balanced_bootstrap_auc(table: pd.DataFrame, seed: int, n: int) -> list[float]:
    rng = np.random.default_rng(seed)
    direct = table[table["law_orientation"] == "direct"]
    inverse = table[table["law_orientation"] == "inverse"]
    values = np.empty(n)
    for index in range(n):
        sample = pd.concat(
            [
                direct.sample(len(direct), replace=True, random_state=int(rng.integers(2**31))),
                inverse.sample(len(inverse), replace=True, random_state=int(rng.integers(2**31))),
            ]
        )
        values[index] = roc_auc_score(
            sample["law_orientation"] == "direct", sample["mean_contrast"]
        )
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def exact_balanced_permutation_p(table: pd.DataFrame) -> float:
    scores = table["mean_contrast"].to_numpy()
    n_direct = int(np.sum(table["law_orientation"] == "direct"))
    observed = roc_auc_score(table["law_orientation"] == "direct", scores)
    distance = abs(observed - 0.5)
    extreme = 0
    total = 0
    for selected in itertools.combinations(range(len(scores)), n_direct):
        labels = np.zeros(len(scores), dtype=bool)
        labels[list(selected)] = True
        auc = roc_auc_score(labels, scores)
        extreme += abs(auc - 0.5) >= distance - 1e-15
        total += 1
    if total != comb(len(scores), n_direct):
        raise RuntimeError("permutation count mismatch")
    return float(extreme / total)


def tfidf_scores(dev: dict, fresh: dict, method: str) -> np.ndarray:
    kwargs = (
        {"lowercase": True, "ngram_range": (1, 2), "sublinear_tf": True}
        if method == "word"
        else {
            "lowercase": True,
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "max_features": 30000,
            "sublinear_tf": True,
        }
    )
    vectorizer = TfidfVectorizer(**kwargs)
    dx = normalize(vectorizer.fit_transform([x["user"] for x in dev["prompts"]]).toarray())
    fx = normalize(vectorizer.transform([x["user"] for x in fresh["prompts"]]).toarray())
    dm = arrays(dev)
    d, midpoint = fit_centroid(dx, dm["physical"])
    return project(fx, d, midpoint)


def orientation_auc(table: pd.DataFrame, column: str = "mean_contrast") -> float:
    return float(
        roc_auc_score(table["law_orientation"] == "direct", table[column])
    )


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    dev, fresh = load(DEV), load(OUT)
    for name in ("development_manifest", "development_states"):
        record = protocol["inputs"][name]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    dm, fm = arrays(dev), arrays(fresh)
    layer = int(protocol["frozen_lens"]["layer"])
    position = protocol["frozen_lens"]["position"]
    dpi = int(np.flatnonzero(dev["positions"] == position)[0])
    fpi = int(np.flatnonzero(fresh["positions"] == position)[0])
    dli = int(np.flatnonzero(dev["layers"] == layer)[0])
    fli = int(np.flatnonzero(fresh["layers"] == layer)[0])
    dx = normalize(dev["states"][dpi, :, dli])
    fx = normalize(fresh["states"][fpi, :, fli])
    direction, midpoint = fit_centroid(dx, dm["physical"])
    hidden_scores = project(fx, direction, midpoint)
    pairs = contrasts(hidden_scores, fm)
    laws = law_table(pairs)

    # Output-head and raw-text controls use the identical paired subtraction.
    output_scores = np.asarray(
        [x["higher_minus_lower_log_odds"] for x in fresh["clean"]], dtype=float
    )
    control_tables = {}
    for name, scores in {
        "output_head": output_scores,
        "word_tfidf": tfidf_scores(dev, fresh, "word"),
        "character_tfidf": tfidf_scores(dev, fresh, "character"),
    }.items():
        control_tables[name] = law_table(contrasts(scores, fm))

    surface_auc = {
        "a": float(
            roc_auc_score(
                laws["law_orientation"] == "direct", laws["surface_a_contrast"]
            )
        ),
        "b": float(
            roc_auc_score(
                laws["law_orientation"] == "direct", laws["surface_b_contrast"]
            )
        ),
    }
    primary_auc = orientation_auc(laws)
    sign_accuracy = float(
        np.mean(
            (laws["mean_contrast"] > 0)
            == (laws["law_orientation"] == "direct")
        )
    )
    controls = {
        name: {
            "orientation_auc": orientation_auc(table),
            "sign_accuracy": float(
                np.mean(
                    (table["mean_contrast"] > 0)
                    == (table["law_orientation"] == "direct")
                )
            ),
        }
        for name, table in control_tables.items()
    }
    summary = {
        "study_id": protocol["study_id"],
        "analysis_sha256": sha256(Path(__file__).resolve()),
        "primary": {
            "layer": layer,
            "position": position,
            "law_orientation_auc": primary_auc,
            "balanced_law_bootstrap_95": balanced_bootstrap_auc(
                laws,
                protocol["inference"]["bootstrap_seed"],
                protocol["inference"]["bootstrap_resamples"],
            ),
            "exact_balanced_permutation_p": exact_balanced_permutation_p(laws),
            "sign_accuracy": sign_accuracy,
            "correct_laws": int(
                np.sum(
                    (laws["mean_contrast"] > 0)
                    == (laws["law_orientation"] == "direct")
                )
            ),
            "laws_total": int(len(laws)),
            "surface_orientation_auc": surface_auc,
            "direct_mean_contrast": float(
                laws.loc[laws["law_orientation"] == "direct", "mean_contrast"].mean()
            ),
            "inverse_mean_contrast": float(
                laws.loc[laws["law_orientation"] == "inverse", "mean_contrast"].mean()
            ),
        },
        "controls": controls,
    }
    rule = protocol["success_rule"]
    summary["passed_all_frozen_criteria"] = bool(
        primary_auc >= rule["law_orientation_auc_minimum"]
        and sign_accuracy >= rule["sign_accuracy_minimum"]
        and min(surface_auc.values()) >= rule["both_surface_auc_minimum"]
    )
    pairs.to_csv(OUT / "matched_pair_contrasts.csv", index=False)
    laws.to_csv(OUT / "law_orientation_scores.csv", index=False)
    for name, table in control_tables.items():
        table.to_csv(OUT / f"{name}_law_scores.csv", index=False)
    (OUT / "statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    # Descriptive layer profile: every layer direction is fit only on old laws.
    profile = []
    for layer_value in dev["layers"]:
        di = int(np.flatnonzero(dev["layers"] == layer_value)[0])
        fi = int(np.flatnonzero(fresh["layers"] == layer_value)[0])
        d, mid = fit_centroid(
            normalize(dev["states"][dpi, :, di]), dm["physical"]
        )
        scores = project(normalize(fresh["states"][fpi, :, fi]), d, mid)
        table = law_table(contrasts(scores, fm))
        profile.append(
            {
                "layer": int(layer_value),
                "orientation_auc": orientation_auc(table),
                "sign_accuracy": float(
                    np.mean(
                        (table["mean_contrast"] > 0)
                        == (table["law_orientation"] == "direct")
                    )
                ),
            }
        )
    profile = pd.DataFrame(profile)
    profile.to_csv(OUT / "descriptive_layer_profile.csv", index=False)

    plt.rcParams.update(
        {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    axes[0].plot(profile["layer"], profile["orientation_auc"], color="#276b9a", lw=2)
    axes[0].axhline(0.5, color="0.5", ls="--", lw=1)
    axes[0].axvline(layer, color="#b33f4a", ls=":", lw=1.5)
    axes[0].scatter([layer], [primary_auc], color="#b33f4a", zorder=3)
    axes[0].set(xlabel="Layer", ylabel="Direct-versus-inverse AUC", ylim=(0, 1.03))
    axes[0].text(0.02, 0.96, "A", transform=axes[0].transAxes, va="top", fontweight="bold")

    ordered = laws.sort_values("mean_contrast")
    colors = ordered["law_orientation"].map({"direct": "#2a9d8f", "inverse": "#7b5ea7"})
    axes[1].barh(range(len(ordered)), ordered["mean_contrast"], color=colors)
    axes[1].axvline(0, color="0.35", lw=1)
    axes[1].set_yticks(
        range(len(ordered)), ordered["law_id"].str.replace("-", " "), fontsize=6.2
    )
    axes[1].set(xlabel="Matched up-minus-down physical score")
    axes[1].text(0.02, 0.96, "B", transform=axes[1].transAxes, va="top", fontweight="bold")

    names = [
        "Hidden\ncontrast",
        "Explicit\nsurface",
        "Rearranged\nsurface",
        "Output\nhead",
        "Word\nTF-IDF",
        "Character\nTF-IDF",
    ]
    vals = [
        primary_auc,
        surface_auc["a"],
        surface_auc["b"],
        controls["output_head"]["orientation_auc"],
        controls["word_tfidf"]["orientation_auc"],
        controls["character_tfidf"]["orientation_auc"],
    ]
    axes[2].bar(range(len(vals)), vals, color=["#276b9a", "#5c8fb6", "#5c8fb6", "#d39b35", "0.65", "0.45"])
    axes[2].axhline(0.5, color="0.5", ls="--", lw=1)
    axes[2].set_xticks(
        range(len(vals)), names, fontsize=6.2, rotation=32, ha="right"
    )
    axes[2].set(ylabel="Direct-versus-inverse AUC", ylim=(0, 1.03))
    axes[2].text(0.02, 0.96, "C", transform=axes[2].transAxes, va="top", fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "relational-contrast-confirmation.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "figures" / "relational-contrast-confirmation.pdf", bbox_inches="tight")
    plt.close(fig)

    p = summary["primary"]
    (OUT / "REPORT.md").write_text(
        "# Relational physical contrast confirmation\n\n"
        f"The frozen layer-{layer} lens classified direct versus inverse "
        f"orientation across {p['laws_total']} entirely new laws with AUC "
        f"{p['law_orientation_auc']:.3f} (balanced-law bootstrap 95% CI "
        f"{p['balanced_law_bootstrap_95'][0]:.3f}–"
        f"{p['balanced_law_bootstrap_95'][1]:.3f}; exact balanced-label "
        f"permutation p={p['exact_balanced_permutation_p']:.5g}). The fixed "
        f"zero threshold classified {p['correct_laws']}/{p['laws_total']} laws "
        f"correctly. Surface-specific AUCs were {surface_auc['a']:.3f} and "
        f"{surface_auc['b']:.3f}. Passed all frozen criteria: "
        f"{summary['passed_all_frozen_criteria']}.\n\n"
        "The readout is relational: it does not decode law type from one state. "
        "It asks how the same hidden physical-outcome score changes when only "
        "the numerical control direction is reversed. Matched subtraction "
        "removes the law name, material, equation wording, and answer order. "
        "A positive contrast denotes a direct law; a negative contrast denotes "
        "an inverse law.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
