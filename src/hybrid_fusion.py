from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ArrayDict = Mapping[str, np.ndarray]


@dataclass
class WeightedFusionRun:
    weights: Dict[str, float]
    params: Dict[str, Any]
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    model: Any


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    labels = np.asarray(y_true)
    preds = np.asarray(y_pred)
    precision_pc = precision_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    recall_pc = recall_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    f1_pc = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "precision_per_class": precision_pc.tolist(),
        "recall_per_class": recall_pc.tolist(),
        "f1_per_class": f1_pc.tolist(),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
        "y_pred": preds.astype(int).tolist(),
    }


def simplex_weight_grid(
    names: Sequence[str],
    step: float = 0.05,
    minimum_weight: float = 0.0,
) -> Iterator[Dict[str, float]]:
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    if minimum_weight < 0 or minimum_weight >= 1:
        raise ValueError("minimum_weight must be in [0, 1)")

    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly, e.g. 0.1 or 0.05")

    min_units = round(minimum_weight / step)
    if not np.isclose(min_units * step, minimum_weight):
        raise ValueError("minimum_weight must align with step")

    n = len(names)
    if n == 0:
        raise ValueError("names must not be empty")

    def recurse(index: int, remaining: int) -> Iterator[list[int]]:
        if index == n - 1:
            if remaining >= min_units:
                yield [remaining]
            return

        max_units = remaining - min_units * (n - index - 1)
        for current in range(min_units, max_units + 1):
            for tail in recurse(index + 1, remaining - current):
                yield [current, *tail]

    for allocation in recurse(0, units):
        yield {name: round(unit * step, 10) for name, unit in zip(names, allocation)}


def build_weighted_features(
    feature_blocks: Mapping[str, ArrayDict],
    weights: Mapping[str, float],
    normalize_by_weight_sum: bool = False,
) -> Dict[str, np.ndarray]:
    block_names = list(feature_blocks.keys())
    splits = list(next(iter(feature_blocks.values())).keys())

    for block_name, split_arrays in feature_blocks.items():
        if set(split_arrays.keys()) != set(splits):
            raise ValueError(f"inconsistent splits for block {block_name}")
        if block_name not in weights:
            raise ValueError(f"missing weight for block {block_name}")

    denom = sum(weights.values()) if normalize_by_weight_sum else 1.0
    if denom <= 0:
        raise ValueError("sum of weights must be positive")

    weighted: Dict[str, np.ndarray] = {}
    for split in splits:
        parts = []
        for block_name in block_names:
            array = np.asarray(feature_blocks[block_name][split], dtype=np.float32)
            parts.append(array * (weights[block_name] / denom))
        weighted[split] = np.hstack(parts)
    return weighted


def search_weighted_feature_fusion(
    feature_blocks: Mapping[str, ArrayDict],
    labels: Mapping[str, Sequence[int]],
    train_model: Callable[[Dict[str, np.ndarray], Dict[str, Any]], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
    weight_candidates: Iterable[Mapping[str, float]] | None = None,
    param_grid: Iterable[Dict[str, Any]] | None = None,
    selection_metric: str = "f1_macro",
    normalize_by_weight_sum: bool = False,
) -> WeightedFusionRun:
    if predict_fn is None:
        predict_fn = lambda model, features: model.predict(features)
    if weight_candidates is None:
        weight_candidates = simplex_weight_grid(list(feature_blocks.keys()))
    if param_grid is None:
        param_grid = [{}]

    best_run: WeightedFusionRun | None = None

    for weights in weight_candidates:
        fused = build_weighted_features(
            feature_blocks,
            weights,
            normalize_by_weight_sum=normalize_by_weight_sum,
        )
        for params in param_grid:
            model = train_model(fused, dict(params))
            train_pred = np.asarray(predict_fn(model, fused["train"]))
            val_pred = np.asarray(predict_fn(model, fused["val"]))
            test_pred = np.asarray(predict_fn(model, fused["test"]))

            run = WeightedFusionRun(
                weights=dict(weights),
                params=dict(params),
                train_metrics=evaluate_predictions(labels["train"], train_pred),
                val_metrics=evaluate_predictions(labels["val"], val_pred),
                test_metrics=evaluate_predictions(labels["test"], test_pred),
                model=model,
            )

            if best_run is None or run.val_metrics[selection_metric] > best_run.val_metrics[selection_metric]:
                best_run = run

    if best_run is None:
        raise RuntimeError("no weighted fusion run was evaluated")
    return best_run


def blend_probabilities(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    normalize_by_weight_sum: bool = True,
) -> np.ndarray:
    names = list(probabilities.keys())
    if set(names) != set(weights.keys()):
        raise ValueError("weights must match probability keys")

    denom = sum(weights.values()) if normalize_by_weight_sum else 1.0
    if denom <= 0:
        raise ValueError("sum of weights must be positive")

    stacked = [
        np.asarray(probabilities[name], dtype=np.float32) * (weights[name] / denom)
        for name in names
    ]
    return np.sum(stacked, axis=0)


class LearnableModalityGate(nn.Module):
    def __init__(self, num_modalities: int, init_logits: Sequence[float] | None = None):
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be positive")
        if init_logits is None:
            init_logits = [0.0] * num_modalities
        if len(init_logits) != num_modalities:
            raise ValueError("init_logits length must equal num_modalities")
        self.logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float))

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, *modalities: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if len(modalities) != self.logits.numel():
            raise ValueError("number of modalities must match gate size")
        weights = self.normalized_weights()
        scaled = [tensor * weights[idx] for idx, tensor in enumerate(modalities)]
        return scaled, weights
