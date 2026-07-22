#!/usr/bin/env python3
"""Whole-mechanism-held-out, gauge-invariant relation GIN.

Executes the relation-GIN portion of frozen Study 4C in
``experiments/graph-isomorphism-generalization-2026-07-18/GAUGE_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from run_graph_isomorphism_gin import (
    BASE_SEED,
    MANIFEST,
    METHODS,
    PROTOCOL,
    STATES,
    GINBlock,
    GraphExample,
    graph_examples,
    index_set,
    load_data,
    plus_one_signflip_p,
    shuffled_adjacency,
    standardize,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
GAUGE_PROTOCOL = OUT / "gauge_protocol.json"
GAUGE_PROTOCOL_MD = OUT / "GAUGE_PROTOCOL.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(value: object) -> object:
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def relation_features(
    examples: Sequence[GraphExample],
) -> list[np.ndarray]:
    # Frozen exclusions remove surface-variant one-hot (columns 1:4) and
    # numeric direction (column 4).  Remaining columns are constant, depth,
    # in/out degree, and in/out strength.
    columns = [0, 5, 6, 7, 8, 9]
    return [
        example.topology_features[:, columns].copy()
        for example in examples
    ]


def local_candidate_pairs(example: GraphExample) -> np.ndarray:
    nodes = list(range(12))
    graph = example.graph
    pairs = [
        (source, target)
        for source in nodes
        for target in nodes
        if graph.nodes[source]["variant"]
        != graph.nodes[target]["variant"]
        and graph.nodes[source]["case"] != graph.nodes[target]["case"]
    ]
    result = np.asarray(pairs, dtype=np.int64)
    if result.shape != (72, 2):
        raise RuntimeError(f"unexpected relation pair shape: {result.shape}")
    return result


def pair_relation_labels(
    node_labels: Sequence[np.ndarray],
    pair_lists: Sequence[np.ndarray],
) -> list[np.ndarray]:
    return [
        (labels[pairs[:, 0]] == labels[pairs[:, 1]]).astype(
            np.float32
        )
        for labels, pairs in zip(node_labels, pair_lists)
    ]


class RelationGIN(torch.nn.Module):
    def __init__(self, input_dim: int, *, use_graph: bool) -> None:
        super().__init__()
        width = 32
        self.use_graph = use_graph
        self.input = torch.nn.Linear(input_dim, width)
        self.blocks = torch.nn.ModuleList(
            [GINBlock(width, 0.10) for _ in range(3)]
        )
        self.node_mlp = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(width, width),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.10),
                )
                for _ in range(3)
            ]
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * width, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(width, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.relu(self.input(features))
        if self.use_graph:
            for block in self.blocks:
                hidden = block(hidden, adjacency)
        else:
            for block in self.node_mlp:
                hidden = hidden + block(hidden)
        batch = torch.arange(
            hidden.shape[0], device=hidden.device
        )[:, None]
        first = hidden[batch, pairs[:, :, 0]]
        second = hidden[batch, pairs[:, :, 1]]
        symmetric = torch.cat(
            [first * second, torch.abs(first - second)], dim=-1
        )
        return self.head(symmetric).squeeze(-1)


def tensor_stack(
    arrays: Sequence[np.ndarray], device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(
        np.stack(arrays), dtype=torch.float32, device=device
    )


def permute_graph_batch(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    pairs: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features = []
    out_adjacency = []
    out_pairs = []
    for index in range(features.shape[0]):
        order = torch.randperm(
            features.shape[1], generator=generator, device=features.device
        )
        inverse = torch.argsort(order)
        out_features.append(features[index, order])
        out_adjacency.append(adjacency[index][order][:, order])
        out_pairs.append(inverse[pairs[index]])
    return (
        torch.stack(out_features),
        torch.stack(out_adjacency),
        torch.stack(out_pairs),
    )


def validation_loss(
    model: RelationGIN,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(features, adjacency, pairs)
        return float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            ).item()
        )


def train_one(
    *,
    model_kind: str,
    feature_list: list[np.ndarray],
    adjacency_list: list[np.ndarray],
    pair_lists: list[np.ndarray],
    pair_labels: list[np.ndarray],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    features, _, _ = standardize(feature_list, train_indices)

    def tensors(
        indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            tensor_stack([features[index] for index in indices], device),
            tensor_stack(
                [adjacency_list[index] for index in indices], device
            ),
            torch.as_tensor(
                np.stack([pair_lists[index] for index in indices]),
                dtype=torch.long,
                device=device,
            ),
            tensor_stack(
                [pair_labels[index] for index in indices], device
            ),
        )

    train_x, train_a, train_p, train_y = tensors(train_indices)
    val_x, val_a, val_p, val_y = tensors(validation_indices)
    test_x, test_a, test_p, test_y = tensors(test_indices)
    model = RelationGIN(
        train_x.shape[-1], use_graph=model_kind == "gin"
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.0001
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 8191)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        epoch_x, epoch_a, epoch_p = permute_graph_batch(
            train_x, train_a, train_p, generator
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(epoch_x, epoch_a, epoch_p)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, train_y
        )
        loss.backward()
        optimizer.step()
        current = validation_loss(
            model, val_x, val_a, val_p, val_y
        )
        if current < best_loss - 1e-5:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no relation-GIN checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(test_x, test_a, test_p).cpu().numpy()

        audit_generator = torch.Generator(device=device)
        audit_generator.manual_seed(seed + 104729)
        audit_x, audit_a, audit_p = permute_graph_batch(
            test_x, test_a, test_p, audit_generator
        )
        audit_logits = model(
            audit_x, audit_a, audit_p
        ).cpu().numpy()
    return {
        "logits": logits,
        "audit_logits": audit_logits,
        "labels": test_y.cpu().numpy(),
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "validation_loss": best_loss,
        "equivariance_max_abs_logit": float(
            np.max(np.abs(logits - audit_logits))
        ),
    }


def shuffled_node_labels(
    examples: Sequence[GraphExample],
    true_node_labels: Sequence[np.ndarray],
    outer_training: set[str],
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output = [labels.copy() for labels in true_node_labels]
    for family in sorted(outer_training):
        family_indices = [
            index
            for index, example in enumerate(examples)
            if example.family == family
        ]
        reference = true_node_labels[family_indices[0]].copy()
        rng.shuffle(reference)
        for index in family_indices:
            output[index] = reference.copy()
    return output


def shuffled_edges(
    examples: Sequence[GraphExample], seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result = []
    for example in examples:
        directed = shuffled_adjacency(example.graph, rng)
        result.append(directed)
    return result


def config_specs() -> list[dict[str, str]]:
    specs = [
        {
            "configuration": "relation_jacobian_gin_physical",
            "method": "jacobian",
            "model_kind": "gin",
            "graph_mode": "observed",
            "target": "physical",
            "label_mode": "observed",
        },
        {
            "configuration": "relation_jacobian_mlp_physical",
            "method": "jacobian",
            "model_kind": "mlp",
            "graph_mode": "observed",
            "target": "physical",
            "label_mode": "observed",
        },
        {
            "configuration": "relation_jacobian_gin_edge_shuffle_physical",
            "method": "jacobian",
            "model_kind": "gin",
            "graph_mode": "shuffled",
            "target": "physical",
            "label_mode": "observed",
        },
        {
            "configuration": "relation_jacobian_gin_label_shuffle_physical",
            "method": "jacobian",
            "model_kind": "gin",
            "graph_mode": "observed",
            "target": "physical",
            "label_mode": "shuffled",
        },
        {
            "configuration": "relation_jacobian_gin_numeric",
            "method": "jacobian",
            "model_kind": "gin",
            "graph_mode": "observed",
            "target": "numeric",
            "label_mode": "observed",
        },
    ]
    for method in ("direct", "raw"):
        for model_kind in ("gin", "mlp"):
            specs.append(
                {
                    "configuration": (
                        f"relation_{method}_{model_kind}_physical"
                    ),
                    "method": method,
                    "model_kind": model_kind,
                    "graph_mode": "observed",
                    "target": "physical",
                    "label_mode": "observed",
                }
            )
    return specs


def parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = value.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(item) for item in value.split(",") if item]


def aggregate_band(
    logits: np.ndarray,
    labels: np.ndarray,
    examples: Sequence[GraphExample],
    indices: Sequence[int],
    band_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    positions = [
        position
        for position, example_index in enumerate(indices)
        if bool(band_mask[examples[example_index].layer_index])
    ]
    band_logits = logits[positions]
    band_labels = labels[positions]
    if not np.all(band_labels == band_labels[0]):
        raise RuntimeError("relation labels differ across layers")
    aggregate = np.mean(band_logits, axis=0)
    target = band_labels[0].astype(int)
    auc = float(roc_auc_score(target, aggregate))
    accuracy = float(accuracy_score(target, aggregate >= 0.0))
    return aggregate, target, auc, accuracy


def run(args: argparse.Namespace) -> None:
    start = time.time()
    followup = json.loads(GAUGE_PROTOCOL.read_text())
    if sha256(PROTOCOL) != followup["parent_protocol_sha256"]:
        raise RuntimeError("parent protocol fingerprint mismatch")
    if (
        sha256(STATES)
        != followup["inputs"]["representations_sha256"]
        or sha256(MANIFEST)
        != followup["inputs"]["prompt_manifest_sha256"]
    ):
        raise RuntimeError("frozen input fingerprint mismatch")
    data = load_data()
    families = list(data["family_order"])
    band_mask = np.asarray(data["band_mask"])
    seeds = parse_seeds(args.seeds)
    device = torch.device(args.device)
    specs = config_specs()
    examples_by_method = {
        method: graph_examples(data, method) for method in METHODS
    }
    seed_rows = []
    pair_rows = []
    total = len(specs) * len(families) * len(seeds)
    completed = 0
    for spec in specs:
        examples = examples_by_method[spec["method"]]
        features = relation_features(examples)
        pair_lists = [
            local_candidate_pairs(example) for example in examples
        ]
        physical_nodes = [
            example.physical.copy() for example in examples
        ]
        numeric_nodes = [
            example.numeric.copy() for example in examples
        ]
        for heldout in families:
            outer_training = set(families) - {heldout}
            test_indices = index_set(examples, {heldout})
            for seed_index in seeds:
                # The target name is deliberately excluded so that physical
                # and numeric-relation controls use identical initialization,
                # splits, and optimization noise.  If their labels are
                # identical, their predictions must therefore be identical.
                algorithm_key = ":".join(
                    [
                        spec["method"],
                        spec["model_kind"],
                        spec["graph_mode"],
                        spec["label_mode"],
                    ]
                )
                run_seed = (
                    BASE_SEED
                    + 200_000 * families.index(heldout)
                    + 2_000 * seed_index
                    + sum(ord(char) for char in algorithm_key)
                )
                validation_family = sorted(outer_training)[
                    seed_index % len(outer_training)
                ]
                inner_training = outer_training - {validation_family}
                train_indices = index_set(examples, inner_training)
                validation_indices = index_set(
                    examples, {validation_family}
                )
                if spec["graph_mode"] == "shuffled":
                    adjacency = shuffled_edges(examples, run_seed)
                else:
                    adjacency = [
                        example.adjacency for example in examples
                    ]
                true_nodes = (
                    physical_nodes
                    if spec["target"] == "physical"
                    else numeric_nodes
                )
                if spec["label_mode"] == "shuffled":
                    training_nodes = shuffled_node_labels(
                        examples,
                        true_nodes,
                        outer_training,
                        run_seed,
                    )
                else:
                    training_nodes = [
                        labels.copy() for labels in true_nodes
                    ]
                training_pairs = pair_relation_labels(
                    training_nodes, pair_lists
                )
                trained = train_one(
                    model_kind=spec["model_kind"],
                    feature_list=features,
                    adjacency_list=adjacency,
                    pair_lists=pair_lists,
                    pair_labels=training_pairs,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    test_indices=test_indices,
                    seed=run_seed,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    device=device,
                )
                true_pairs = pair_relation_labels(
                    true_nodes, pair_lists
                )
                test_true = np.stack(
                    [true_pairs[index] for index in test_indices]
                )
                (
                    aggregate,
                    target,
                    auc,
                    accuracy,
                ) = aggregate_band(
                    np.asarray(trained["logits"]),
                    test_true,
                    examples,
                    test_indices,
                    band_mask,
                )
                audit_aggregate, _, audit_auc, _ = aggregate_band(
                    np.asarray(trained["audit_logits"]),
                    test_true,
                    examples,
                    test_indices,
                    band_mask,
                )
                seed_rows.append(
                    {
                        **spec,
                        "heldout_family": heldout,
                        "seed_index": seed_index,
                        "run_seed": run_seed,
                        "validation_family": validation_family,
                        "auc": auc,
                        "accuracy": accuracy,
                        "audit_auc": audit_auc,
                        "audit_auc_loss": auc - audit_auc,
                        "equivariance_max_abs_logit": trained[
                            "equivariance_max_abs_logit"
                        ],
                        "best_epoch": trained["best_epoch"],
                        "epochs_run": trained["epochs_run"],
                        "validation_loss": trained["validation_loss"],
                    }
                )
                reference_pairs = pair_lists[test_indices[0]]
                for pair_index, (
                    pair,
                    logit,
                    audit_logit,
                    label,
                ) in enumerate(
                    zip(
                        reference_pairs,
                        aggregate,
                        audit_aggregate,
                        target,
                    )
                ):
                    pair_rows.append(
                        {
                            **spec,
                            "heldout_family": heldout,
                            "seed_index": seed_index,
                            "pair_index": pair_index,
                            "source_node": int(pair[0]),
                            "target_node": int(pair[1]),
                            "label": int(label),
                            "band_mean_logit": float(logit),
                            "audit_band_mean_logit": float(
                                audit_logit
                            ),
                        }
                    )
                completed += 1
                if completed % max(1, len(seeds)) == 0:
                    print(
                        f"[{completed}/{total}] "
                        f"{spec['configuration']} heldout={heldout} "
                        f"elapsed={time.time() - start:.1f}s",
                        flush=True,
                    )

    seed_frame = pd.DataFrame(seed_rows)
    pair_frame = pd.DataFrame(pair_rows)
    seed_frame.to_csv(
        OUT / "relation_gin_seed_results.csv", index=False
    )
    pair_frame.to_csv(
        OUT / "relation_gin_pair_logits.csv", index=False
    )
    group_columns = [
        "configuration",
        "method",
        "model_kind",
        "graph_mode",
        "target",
        "label_mode",
        "heldout_family",
        "pair_index",
        "source_node",
        "target_node",
        "label",
    ]
    ensemble = (
        pair_frame.groupby(group_columns, as_index=False)[
            ["band_mean_logit", "audit_band_mean_logit"]
        ]
        .mean()
    )
    family_rows = []
    for (configuration, family), frame in ensemble.groupby(
        ["configuration", "heldout_family"], sort=True
    ):
        labels = frame["label"].to_numpy(int)
        logits = frame["band_mean_logit"].to_numpy(float)
        audit_logits = frame[
            "audit_band_mean_logit"
        ].to_numpy(float)
        family_rows.append(
            {
                "configuration": configuration,
                "heldout_family": family,
                "auc": float(roc_auc_score(labels, logits)),
                "accuracy": float(
                    accuracy_score(labels, logits >= 0.0)
                ),
                "audit_auc": float(
                    roc_auc_score(labels, audit_logits)
                ),
            }
        )
    family_metrics = pd.DataFrame(family_rows)
    family_metrics["audit_auc_loss"] = (
        family_metrics["auc"] - family_metrics["audit_auc"]
    )
    family_metrics.to_csv(
        OUT / "relation_gin_family_metrics.csv", index=False
    )
    ensemble.to_csv(
        OUT / "relation_gin_ensemble_pair_logits.csv", index=False
    )
    pivot = family_metrics.pivot(
        index="heldout_family",
        columns="configuration",
        values="auc",
    )
    primary = pivot["relation_jacobian_gin_physical"].reindex(families)
    gin_minus_mlp = (
        primary
        - pivot["relation_jacobian_mlp_physical"].reindex(families)
    )
    gin_minus_shuffle = (
        primary
        - pivot[
            "relation_jacobian_gin_edge_shuffle_physical"
        ].reindex(families)
    )
    numeric_contrast = (
        primary
        - pivot["relation_jacobian_gin_numeric"].reindex(families)
    )
    audit = (
        family_metrics[
            family_metrics["configuration"]
            == "relation_jacobian_gin_physical"
        ]
        .set_index("heldout_family")["audit_auc"]
        .reindex(families)
    )
    strong = bool(
        np.sum(primary.to_numpy() > 0.5) == 6
        and np.sum(gin_minus_mlp.to_numpy() > 0) == 6
        and np.sum(gin_minus_shuffle.to_numpy() > 0) == 6
        and plus_one_signflip_p(gin_minus_mlp) <= 0.05
        and plus_one_signflip_p(gin_minus_shuffle) <= 0.05
        and np.max(primary.to_numpy() - audit.to_numpy()) <= 0.02
    )
    summary = {
        "study_id": followup["study_id"],
        "gauge_protocol_sha256": sha256(GAUGE_PROTOCOL),
        "gauge_protocol_markdown_sha256": sha256(
            GAUGE_PROTOCOL_MD
        ),
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "primary_family_auc": primary.to_dict(),
        "gin_minus_pair_mlp": gin_minus_mlp.to_dict(),
        "gin_minus_edge_shuffle": gin_minus_shuffle.to_dict(),
        "physical_minus_numeric_relation": numeric_contrast.to_dict(),
        "primary_families_above_half": int(
            np.sum(primary.to_numpy() > 0.5)
        ),
        "gin_minus_mlp_positive_families": int(
            np.sum(gin_minus_mlp.to_numpy() > 0)
        ),
        "gin_minus_shuffle_positive_families": int(
            np.sum(gin_minus_shuffle.to_numpy() > 0)
        ),
        "gin_minus_mlp_plus_one_signflip_p": plus_one_signflip_p(
            gin_minus_mlp
        ),
        "gin_minus_shuffle_plus_one_signflip_p": (
            plus_one_signflip_p(gin_minus_shuffle)
        ),
        "max_permutation_audit_auc_loss": float(
            np.max(primary.to_numpy() - audit.to_numpy())
        ),
        "physical_numeric_max_absolute_difference": float(
            np.max(np.abs(numeric_contrast.to_numpy()))
        ),
        "strong_relation_gin_gate": strong,
        "family_metrics": family_metrics.to_dict(orient="records"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "elapsed_seconds": float(time.time() - start),
    }
    (OUT / "study4c_relation_gin_statistics.json").write_text(
        json.dumps(safe_json(summary), indent=2) + "\n"
    )
    print(json.dumps(safe_json(summary), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:20")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
