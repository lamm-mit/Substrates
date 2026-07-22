#!/usr/bin/env python3
"""Whole-mechanism-held-out GIN experiment from the frozen graph protocol.

This script executes Study 3 in
``experiments/graph-isomorphism-generalization-2026-07-18/PROTOCOL.md``.
It uses archived hidden states only and never runs Gemma.
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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "experiments"
    / "graph-isomorphism-generalization-2026-07-18"
)
PROTOCOL = OUT / "protocol.json"
STATES = (
    ROOT
    / "experiments"
    / "option-free-question-end-2026-07-18"
    / "representations.npz"
)
MANIFEST = (
    ROOT
    / "experiments"
    / "late-physics-representation-replication-2026-07-17"
    / "prompt_manifest.json"
)
VARIANTS = ("anchor", "physics_paraphrase", "lexical_counterfactual")
METHODS = ("jacobian", "direct", "raw")
BASE_SEED = 20260718


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


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise FloatingPointError("zero-norm state")
    return values / norms


def cosine_layers(values: np.ndarray) -> np.ndarray:
    unit = normalize(values.astype(np.float64))
    return np.einsum("ild,jld->lij", unit, unit, optimize=True)


def plus_one_signflip_p(values: Sequence[float]) -> float:
    data = np.asarray(values, dtype=float)
    observed = float(np.mean(data))
    signs = np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=len(data)))
    )
    null = np.mean(signs * data[None, :], axis=1)
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + len(null)))


def load_data() -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text())
    expected_states = protocol["inputs"]["representations"]["sha256"]
    expected_manifest = protocol["inputs"]["prompt_manifest"]["sha256"]
    if sha256(STATES) != expected_states or sha256(MANIFEST) != expected_manifest:
        raise RuntimeError("frozen input fingerprint mismatch")

    manifest = json.loads(MANIFEST.read_text())
    metadata = {
        str(row["prompt_id"]): row for row in manifest["prompts"]
    }
    with np.load(STATES, allow_pickle=False) as archive:
        prompt_ids = archive["prompt_ids"].astype(str)
        layers = archive["layers"].astype(int)
        raw = archive["raw_states"][0].astype(np.float32)
        direct = archive["direct_decoder_basis"][0].astype(np.float32)
        jacobian_fits = archive["jacobian_decoder_basis"][
            :, 0
        ].astype(np.float32)
    rows = []
    for prompt_id in prompt_ids:
        key = prompt_id.removesuffix("--answer-code")
        rows.append(metadata[key])
    families = np.asarray([str(row["family_id"]) for row in rows])
    cases = np.asarray([str(row["case_id"]) for row in rows])
    variants = np.asarray([str(row["variant"]) for row in rows])
    physical = np.asarray(
        [
            bool(row["expected_outcome"] == row["outcome_positive"])
            for row in rows
        ],
        dtype=np.int64,
    )
    numeric = np.asarray(
        [bool(row["numeric_direction"] == "increase") for row in rows],
        dtype=np.int64,
    )
    states = {
        "jacobian": np.mean(jacobian_fits, axis=0),
        "direct": direct,
        "raw": raw,
    }
    similarities = {
        method: cosine_layers(values)
        for method, values in states.items()
    }
    depths = layers / 41.0 * 100.0
    return {
        "prompt_ids": prompt_ids,
        "families": families,
        "cases": cases,
        "variants": variants,
        "physical": physical,
        "numeric": numeric,
        "family_order": sorted(set(families)),
        "layers": layers,
        "depths": depths,
        "band_mask": (depths >= 38.0) & (depths <= 92.0),
        "states": states,
        "similarities": similarities,
    }


def selected_family_graph(
    similarity: np.ndarray,
    family: str,
    data: Mapping[str, object],
) -> nx.DiGraph:
    families = np.asarray(data["families"])
    cases = np.asarray(data["cases"])
    variants = np.asarray(data["variants"])
    physical = np.asarray(data["physical"])
    numeric = np.asarray(data["numeric"])
    global_indices = np.flatnonzero(families == family)
    local = {
        int(global_index): i
        for i, global_index in enumerate(global_indices)
    }
    graph = nx.DiGraph()
    for global_index in global_indices:
        graph.add_node(
            local[int(global_index)],
            global_index=int(global_index),
            variant=str(variants[global_index]),
            case=str(cases[global_index]),
            physical=int(physical[global_index]),
            numeric=int(numeric[global_index]),
        )
    for global_source in global_indices:
        for target_variant in VARIANTS:
            if target_variant == variants[global_source]:
                continue
            candidates = global_indices[
                (variants[global_indices] == target_variant)
                & (cases[global_indices] != cases[global_source])
            ]
            target = min(
                candidates.tolist(),
                key=lambda index: (
                    -float(similarity[global_source, index]),
                    int(index),
                ),
            )
            graph.add_edge(
                local[int(global_source)],
                local[int(target)],
                weight=float(similarity[global_source, target]),
            )
    if graph.number_of_nodes() != 12 or graph.number_of_edges() != 24:
        raise RuntimeError("unexpected graph dimensions")
    return graph


def bidirectional_adjacency(graph: nx.DiGraph) -> np.ndarray:
    adjacency = np.zeros((12, 12), dtype=np.float32)
    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes["weight"])
        adjacency[source, target] += weight
        adjacency[target, source] += weight
    return adjacency


def shuffled_adjacency(
    graph: nx.DiGraph, rng: np.random.Generator
) -> np.ndarray:
    edges = []
    for source, attributes in graph.nodes(data=True):
        for target_variant in VARIANTS:
            if target_variant == attributes["variant"]:
                continue
            candidates = [
                node
                for node, target_attributes in graph.nodes(data=True)
                if target_attributes["variant"] == target_variant
                and target_attributes["case"] != attributes["case"]
            ]
            edges.append((source, int(rng.choice(candidates))))
    weights = np.asarray(
        [float(attributes["weight"]) for *_, attributes in graph.edges(data=True)]
    )
    rng.shuffle(weights)
    adjacency = np.zeros((12, 12), dtype=np.float32)
    for (source, target), weight in zip(edges, weights):
        adjacency[source, target] += float(weight)
        adjacency[target, source] += float(weight)
    return adjacency


@dataclass
class GraphExample:
    family: str
    method: str
    layer_index: int
    layer: int
    depth: float
    node_global_indices: np.ndarray
    topology_features: np.ndarray
    adjacency: np.ndarray
    physical: np.ndarray
    numeric: np.ndarray
    graph: nx.DiGraph


def graph_examples(
    data: Mapping[str, object], method: str
) -> list[GraphExample]:
    layers = np.asarray(data["layers"])
    depths = np.asarray(data["depths"])
    similarity_layers = np.asarray(data["similarities"][method])
    examples = []
    for family in data["family_order"]:
        for layer_index, (layer, depth) in enumerate(zip(layers, depths)):
            graph = selected_family_graph(
                similarity_layers[layer_index], str(family), data
            )
            nodes = sorted(graph.nodes)
            global_indices = np.asarray(
                [graph.nodes[node]["global_index"] for node in nodes],
                dtype=np.int64,
            )
            variants = [
                str(graph.nodes[node]["variant"]) for node in nodes
            ]
            variant_onehot = np.asarray(
                [
                    [float(value == variant) for value in VARIANTS]
                    for variant in variants
                ],
                dtype=np.float32,
            )
            in_degree = np.asarray(
                [graph.in_degree(node) for node in nodes],
                dtype=np.float32,
            )
            out_degree = np.asarray(
                [graph.out_degree(node) for node in nodes],
                dtype=np.float32,
            )
            in_strength = np.asarray(
                [
                    sum(
                        float(attributes["weight"])
                        for *_, attributes in graph.in_edges(
                            node, data=True
                        )
                    )
                    for node in nodes
                ],
                dtype=np.float32,
            )
            out_strength = np.asarray(
                [
                    sum(
                        float(attributes["weight"])
                        for *_, attributes in graph.out_edges(
                            node, data=True
                        )
                    )
                    for node in nodes
                ],
                dtype=np.float32,
            )
            numeric_sign = np.asarray(
                [
                    1.0 if graph.nodes[node]["numeric"] else -1.0
                    for node in nodes
                ],
                dtype=np.float32,
            )
            topology = np.column_stack(
                [
                    np.ones(12, dtype=np.float32),
                    variant_onehot,
                    numeric_sign,
                    np.full(12, float(depth) / 100.0, dtype=np.float32),
                    in_degree,
                    out_degree,
                    in_strength,
                    out_strength,
                ]
            ).astype(np.float32)
            examples.append(
                GraphExample(
                    family=str(family),
                    method=method,
                    layer_index=layer_index,
                    layer=int(layer),
                    depth=float(depth),
                    node_global_indices=global_indices,
                    topology_features=topology,
                    adjacency=bidirectional_adjacency(graph),
                    physical=np.asarray(
                        [graph.nodes[node]["physical"] for node in nodes],
                        dtype=np.int64,
                    ),
                    numeric=np.asarray(
                        [graph.nodes[node]["numeric"] for node in nodes],
                        dtype=np.int64,
                    ),
                    graph=graph,
                )
            )
    return examples


def fit_fold_pca(
    data: Mapping[str, object],
    method: str,
    training_families: set[str],
) -> tuple[PCA, np.ndarray]:
    # Use float64 for the randomized SVD.  The archived arrays are finite, but
    # the Accelerate-backed float32 matmul used by this scikit-learn build emits
    # spurious overflow warnings during power iteration.
    states = np.asarray(data["states"][method], dtype=np.float64)
    families = np.asarray(data["families"])
    train_indices = np.flatnonzero(
        np.isin(families, sorted(training_families))
    )
    matrix = states[train_indices].transpose(1, 0, 2).reshape(
        -1, states.shape[-1]
    )
    pca = PCA(
        n_components=32,
        svd_solver="randomized",
        random_state=BASE_SEED,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module="sklearn"
        )
        pca.fit(matrix)
    projected = np.empty(
        (states.shape[0], states.shape[1], 32), dtype=np.float32
    )
    for layer_index in range(states.shape[1]):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module="sklearn"
            )
            transformed = pca.transform(states[:, layer_index])
        if not np.all(np.isfinite(transformed)):
            raise FloatingPointError(
                f"non-finite PCA projection for {method}, layer "
                f"{layer_index}"
            )
        projected[:, layer_index] = transformed.astype(np.float32)
    return pca, projected


def add_state_features(
    examples: list[GraphExample], projected: np.ndarray
) -> list[np.ndarray]:
    return [
        np.column_stack(
            [
                example.topology_features,
                projected[
                    example.node_global_indices, example.layer_index
                ],
            ]
        ).astype(np.float32)
        for example in examples
    ]


def topology_feature_list(
    examples: list[GraphExample],
) -> list[np.ndarray]:
    return [example.topology_features.copy() for example in examples]


def standardize(
    feature_list: list[np.ndarray], train_indices: Sequence[int]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    matrix = np.concatenate(
        [feature_list[index] for index in train_indices], axis=0
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    transformed = [
        ((features - mean) / scale).astype(np.float32)
        for features in feature_list
    ]
    return transformed, mean, scale


class GINBlock(torch.nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.epsilon = torch.nn.Parameter(torch.zeros(()))
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, hidden: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        aggregate = torch.bmm(adjacency, hidden)
        update = self.layers((1.0 + self.epsilon) * hidden + aggregate)
        return torch.relu(hidden + self.dropout(update))


class GIN(torch.nn.Module):
    def __init__(self, input_dim: int, width: int = 32) -> None:
        super().__init__()
        self.input = torch.nn.Linear(input_dim, width)
        self.blocks = torch.nn.ModuleList(
            [GINBlock(width, 0.10) for _ in range(3)]
        )
        self.head = torch.nn.Linear(width, 1)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        hidden = torch.relu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden, adjacency)
        return self.head(hidden).squeeze(-1)


class NodeMLP(torch.nn.Module):
    def __init__(self, input_dim: int, width: int = 32) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(width, 1),
        )

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        del adjacency
        return self.layers(features).squeeze(-1)


def tensor_stack(
    arrays: Iterable[np.ndarray], device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(
        np.stack(list(arrays)), dtype=torch.float32, device=device
    )


def permute_batch(
    features: torch.Tensor,
    adjacency: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    permuted_features = []
    permuted_adjacency = []
    permuted_labels = []
    for index in range(features.shape[0]):
        order = torch.randperm(
            features.shape[1], generator=generator, device=features.device
        )
        permuted_features.append(features[index, order])
        permuted_adjacency.append(
            adjacency[index][order][:, order]
        )
        permuted_labels.append(labels[index, order])
    return (
        torch.stack(permuted_features),
        torch.stack(permuted_adjacency),
        torch.stack(permuted_labels),
    )


def evaluate_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(features, adjacency)
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
    labels_list: list[np.ndarray],
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
    train_x = tensor_stack(
        [features[index] for index in train_indices], device
    )
    train_a = tensor_stack(
        [adjacency_list[index] for index in train_indices], device
    )
    train_y = tensor_stack(
        [labels_list[index] for index in train_indices], device
    )
    val_x = tensor_stack(
        [features[index] for index in validation_indices], device
    )
    val_a = tensor_stack(
        [adjacency_list[index] for index in validation_indices], device
    )
    val_y = tensor_stack(
        [labels_list[index] for index in validation_indices], device
    )
    test_x = tensor_stack(
        [features[index] for index in test_indices], device
    )
    test_a = tensor_stack(
        [adjacency_list[index] for index in test_indices], device
    )
    test_y = tensor_stack(
        [labels_list[index] for index in test_indices], device
    )

    model: torch.nn.Module
    if model_kind == "gin":
        model = GIN(train_x.shape[-1])
    elif model_kind == "mlp":
        model = NodeMLP(train_x.shape[-1])
    else:
        raise ValueError(model_kind)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.0001
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 9973)
    for epoch in range(max_epochs):
        model.train()
        epoch_x, epoch_a, epoch_y = permute_batch(
            train_x, train_a, train_y, generator
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(epoch_x, epoch_a)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, epoch_y
        )
        loss.backward()
        optimizer.step()
        validation_loss = evaluate_loss(
            model, val_x, val_a, val_y
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(test_x, test_a).cpu().numpy()

        audit_generator = torch.Generator(device=device)
        audit_generator.manual_seed(seed + 104729)
        audit_x = []
        audit_a = []
        inverse_orders = []
        for index in range(test_x.shape[0]):
            order = torch.randperm(
                test_x.shape[1],
                generator=audit_generator,
                device=device,
            )
            inverse = torch.argsort(order)
            audit_x.append(test_x[index, order])
            audit_a.append(test_a[index][order][:, order])
            inverse_orders.append(inverse)
        audit_logits = model(
            torch.stack(audit_x), torch.stack(audit_a)
        )
        audit_logits = torch.stack(
            [
                audit_logits[index, inverse_orders[index]]
                for index in range(audit_logits.shape[0])
            ]
        ).cpu().numpy()
    return {
        "logits": logits,
        "audit_logits": audit_logits,
        "labels": test_y.cpu().numpy(),
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "epochs_run": epoch + 1,
        "equivariance_max_abs_logit": float(
            np.max(np.abs(logits - audit_logits))
        ),
    }


def shuffled_labels(
    examples: list[GraphExample],
    true_labels: list[np.ndarray],
    training_families: set[str],
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output = [labels.copy() for labels in true_labels]
    for family in sorted(training_families):
        family_indices = [
            index
            for index, example in enumerate(examples)
            if example.family == family
        ]
        reference = true_labels[family_indices[0]].copy()
        rng.shuffle(reference)
        for index in family_indices:
            output[index] = reference.copy()
    return output


def shuffled_edges(
    examples: list[GraphExample], seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        shuffled_adjacency(example.graph, rng) for example in examples
    ]


def index_set(
    examples: list[GraphExample], families: set[str]
) -> list[int]:
    return [
        index
        for index, example in enumerate(examples)
        if example.family in families
    ]


def auc_from_band(
    logits: np.ndarray,
    labels: np.ndarray,
    examples: list[GraphExample],
    indices: Sequence[int],
    band_mask: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    selected_positions = [
        position
        for position, example_index in enumerate(indices)
        if bool(band_mask[examples[example_index].layer_index])
    ]
    band_logits = logits[selected_positions]
    band_labels = labels[selected_positions]
    if not np.all(band_labels == band_labels[0]):
        raise RuntimeError("node labels differ across layers")
    node_logits = np.mean(band_logits, axis=0)
    node_labels = band_labels[0].astype(int)
    auc = float(roc_auc_score(node_labels, node_logits))
    accuracy = float(
        accuracy_score(node_labels, node_logits >= 0.0)
    )
    return auc, accuracy, node_logits, node_labels


def configuration_specs() -> list[dict[str, str]]:
    specs = [
        {
            "configuration": "topology_gin_physical",
            "method": "jacobian",
            "feature_mode": "topology",
            "model_kind": "gin",
            "graph_mode": "observed",
            "target": "physical",
            "label_mode": "observed",
        },
        {
            "configuration": "topology_mlp_physical",
            "method": "jacobian",
            "feature_mode": "topology",
            "model_kind": "mlp",
            "graph_mode": "observed",
            "target": "physical",
            "label_mode": "observed",
        },
        {
            "configuration": "topology_gin_edge_shuffle_physical",
            "method": "jacobian",
            "feature_mode": "topology",
            "model_kind": "gin",
            "graph_mode": "shuffled",
            "target": "physical",
            "label_mode": "observed",
        },
        {
            "configuration": "topology_gin_label_shuffle_physical",
            "method": "jacobian",
            "feature_mode": "topology",
            "model_kind": "gin",
            "graph_mode": "observed",
            "target": "physical",
            "label_mode": "shuffled",
        },
        {
            "configuration": "topology_gin_numeric",
            "method": "jacobian",
            "feature_mode": "topology",
            "model_kind": "gin",
            "graph_mode": "observed",
            "target": "numeric",
            "label_mode": "observed",
        },
    ]
    for method in METHODS:
        for model_kind in ("gin", "mlp"):
            specs.append(
                {
                    "configuration": (
                        f"state_{method}_{model_kind}_physical"
                    ),
                    "method": method,
                    "feature_mode": "state",
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


def run(args: argparse.Namespace) -> None:
    start = time.time()
    data = load_data()
    family_order = list(data["family_order"])
    band_mask = np.asarray(data["band_mask"])
    device = torch.device(args.device)
    seeds = parse_seeds(args.seeds)
    specs = configuration_specs()
    if args.configurations:
        selected = set(args.configurations.split(","))
        specs = [
            spec for spec in specs
            if spec["configuration"] in selected
        ]
    examples_by_method = {
        method: graph_examples(data, method) for method in METHODS
    }
    results_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    pca_cache: dict[tuple[str, str], np.ndarray] = {}
    total = len(specs) * len(family_order) * len(seeds)
    completed = 0

    for spec in specs:
        method = spec["method"]
        examples = examples_by_method[method]
        true_physical = [example.physical for example in examples]
        true_numeric = [example.numeric for example in examples]
        for heldout in family_order:
            outer_training = set(family_order) - {heldout}
            test_indices = index_set(examples, {heldout})
            if spec["feature_mode"] == "state":
                cache_key = (method, heldout)
                if cache_key not in pca_cache:
                    _, projected = fit_fold_pca(
                        data, method, outer_training
                    )
                    pca_cache[cache_key] = projected
                feature_list = add_state_features(
                    examples, pca_cache[cache_key]
                )
            else:
                feature_list = topology_feature_list(examples)

            for seed_index in seeds:
                run_seed = (
                    BASE_SEED
                    + 100_000 * family_order.index(heldout)
                    + 1_000 * seed_index
                    + sum(ord(char) for char in spec["configuration"])
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
                    adjacency_list = shuffled_edges(examples, run_seed)
                else:
                    adjacency_list = [
                        example.adjacency for example in examples
                    ]
                target_labels = (
                    true_physical
                    if spec["target"] == "physical"
                    else true_numeric
                )
                if spec["label_mode"] == "shuffled":
                    training_labels = shuffled_labels(
                        examples,
                        target_labels,
                        outer_training,
                        run_seed,
                    )
                else:
                    training_labels = [
                        labels.copy() for labels in target_labels
                    ]

                trained = train_one(
                    model_kind=spec["model_kind"],
                    feature_list=feature_list,
                    adjacency_list=adjacency_list,
                    labels_list=training_labels,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    test_indices=test_indices,
                    seed=run_seed,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    device=device,
                )
                # Test labels are never shuffled, even for the label-shuffle
                # falsification.
                test_true = np.stack(
                    [target_labels[index] for index in test_indices]
                )
                auc, accuracy, node_logits, node_labels = auc_from_band(
                    np.asarray(trained["logits"]),
                    test_true,
                    examples,
                    test_indices,
                    band_mask,
                )
                audit_auc, _, audit_node_logits, _ = auc_from_band(
                    np.asarray(trained["audit_logits"]),
                    test_true,
                    examples,
                    test_indices,
                    band_mask,
                )
                results_rows.append(
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
                reference_example = examples[test_indices[0]]
                for node_index, (logit, audit_logit, label) in enumerate(
                    zip(node_logits, audit_node_logits, node_labels)
                ):
                    global_index = int(
                        reference_example.node_global_indices[node_index]
                    )
                    node_rows.append(
                        {
                            **spec,
                            "heldout_family": heldout,
                            "seed_index": seed_index,
                            "node_index": node_index,
                            "prompt_id": str(
                                data["prompt_ids"][global_index]
                            ),
                            "label": int(label),
                            "band_mean_logit": float(logit),
                            "audit_band_mean_logit": float(audit_logit),
                        }
                    )
                completed += 1
                if completed % max(1, min(20, len(seeds))) == 0:
                    elapsed = time.time() - start
                    print(
                        f"[{completed}/{total}] "
                        f"{spec['configuration']} heldout={heldout} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    results = pd.DataFrame(results_rows)
    nodes = pd.DataFrame(node_rows)
    results.to_csv(OUT / "gin_seed_results.csv", index=False)
    nodes.to_csv(OUT / "gin_node_logits.csv", index=False)

    # Ensemble seeds before inference.  Each seed contributes one band-mean
    # logit per held-out node.
    ensemble_nodes = (
        nodes.groupby(
            [
                "configuration",
                "method",
                "feature_mode",
                "model_kind",
                "graph_mode",
                "target",
                "label_mode",
                "heldout_family",
                "prompt_id",
                "label",
            ],
            as_index=False,
        )[["band_mean_logit", "audit_band_mean_logit"]]
        .mean()
    )
    family_rows = []
    for keys, frame in ensemble_nodes.groupby(
        ["configuration", "heldout_family"], sort=True
    ):
        configuration, heldout = keys
        labels = frame["label"].to_numpy(int)
        logits = frame["band_mean_logit"].to_numpy(float)
        audit_logits = frame["audit_band_mean_logit"].to_numpy(float)
        family_rows.append(
            {
                "configuration": configuration,
                "heldout_family": heldout,
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
    family_metrics.to_csv(OUT / "gin_family_metrics.csv", index=False)
    ensemble_nodes.to_csv(
        OUT / "gin_ensemble_node_logits.csv", index=False
    )

    pivot = family_metrics.pivot(
        index="heldout_family", columns="configuration", values="auc"
    )
    required_primary = {
        "topology_gin_physical",
        "topology_mlp_physical",
        "topology_gin_edge_shuffle_physical",
        "topology_gin_numeric",
    }
    if not required_primary.issubset(set(pivot.columns)):
        partial = {
            "study_id": json.loads(PROTOCOL.read_text())["study_id"],
            "status": "partial_debug_run",
            "configurations": sorted(pivot.columns.tolist()),
            "seeds": seeds,
            "max_epochs": args.max_epochs,
            "family_metrics": family_metrics.to_dict(orient="records"),
            "elapsed_seconds": float(time.time() - start),
        }
        (OUT / "study3_gin_statistics.json").write_text(
            json.dumps(safe_json(partial), indent=2) + "\n"
        )
        print(json.dumps(safe_json(partial), indent=2))
        return
    primary = pivot["topology_gin_physical"]
    gin_minus_mlp = primary - pivot["topology_mlp_physical"]
    gin_minus_shuffle = (
        primary - pivot["topology_gin_edge_shuffle_physical"]
    )
    physical_minus_numeric = (
        primary - pivot["topology_gin_numeric"]
    )
    primary_audit = (
        family_metrics[
            family_metrics["configuration"] == "topology_gin_physical"
        ]
        .set_index("heldout_family")["audit_auc"]
        .reindex(family_order)
    )
    strong_gate = bool(
        np.sum(primary.reindex(family_order).to_numpy() > 0.5) == 6
        and np.sum(gin_minus_mlp.reindex(family_order).to_numpy() > 0) == 6
        and np.sum(
            gin_minus_shuffle.reindex(family_order).to_numpy() > 0
        ) == 6
        and plus_one_signflip_p(gin_minus_mlp.reindex(family_order)) <= 0.05
        and plus_one_signflip_p(
            gin_minus_shuffle.reindex(family_order)
        ) <= 0.05
        and np.max(
            primary.reindex(family_order).to_numpy()
            - primary_audit.to_numpy()
        )
        <= 0.02
        and np.sum(
            physical_minus_numeric.reindex(family_order).to_numpy() > 0
        )
        >= 4
    )
    summary = {
        "study_id": json.loads(PROTOCOL.read_text())["study_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "input_sha256": {
            "representations": sha256(STATES),
            "prompt_manifest": sha256(MANIFEST),
        },
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "device": str(device),
        "primary_family_auc": primary.reindex(family_order).to_dict(),
        "gin_minus_mlp": gin_minus_mlp.reindex(family_order).to_dict(),
        "gin_minus_edge_shuffle": gin_minus_shuffle.reindex(
            family_order
        ).to_dict(),
        "physical_minus_numeric": physical_minus_numeric.reindex(
            family_order
        ).to_dict(),
        "primary_families_above_half": int(
            np.sum(primary.reindex(family_order).to_numpy() > 0.5)
        ),
        "gin_minus_mlp_positive_families": int(
            np.sum(gin_minus_mlp.reindex(family_order).to_numpy() > 0)
        ),
        "gin_minus_edge_shuffle_positive_families": int(
            np.sum(
                gin_minus_shuffle.reindex(family_order).to_numpy() > 0
            )
        ),
        "gin_minus_mlp_plus_one_signflip_p": plus_one_signflip_p(
            gin_minus_mlp.reindex(family_order)
        ),
        "gin_minus_edge_shuffle_plus_one_signflip_p": (
            plus_one_signflip_p(
                gin_minus_shuffle.reindex(family_order)
            )
        ),
        "max_permutation_audit_auc_loss": float(
            np.max(
                primary.reindex(family_order).to_numpy()
                - primary_audit.to_numpy()
            )
        ),
        "physical_over_numeric_families": int(
            np.sum(
                physical_minus_numeric.reindex(family_order).to_numpy()
                > 0
            )
        ),
        "strong_whole_mechanism_gate": strong_gate,
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
    (OUT / "study3_gin_statistics.json").write_text(
        json.dumps(safe_json(summary), indent=2) + "\n"
    )
    print(json.dumps(safe_json(summary), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default="0:20",
        help="Seed indices as start:stop or comma-separated integers.",
    )
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--configurations",
        default="",
        help="Optional comma-separated configuration names.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
