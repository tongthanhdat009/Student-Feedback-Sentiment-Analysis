from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def find_project_root(start: str | Path | None = None) -> Path:
    if start is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start).resolve()
        if current.is_file():
            current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    raise FileNotFoundError("Project root not found")


def load_experiment_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_aligned_truth(*summaries: Mapping[str, Any], split: str) -> list[int]:
    truths = [list(summary[split]["y_true"]) for summary in summaries]
    if not truths:
        raise ValueError("at least one summary is required")
    first = truths[0]
    for other in truths[1:]:
        if other != first:
            raise ValueError(f"mismatched y_true for split={split}")
    return first


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_map: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    if label_map is None:
        label_map = LABEL_MAP

    labels = sorted(label_map.keys())
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_per_class": precision_pc.tolist(),
        "recall_per_class": recall_pc.tolist(),
        "f1_per_class": f1_pc.tolist(),
        "support_per_class": support_pc.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[label_map[label] for label in labels],
            zero_division=0,
        ),
        "y_true": list(map(int, y_true)),
        "y_pred": list(map(int, y_pred)),
    }


def neutral_aware_ensemble(
    primary_preds: Sequence[int],
    secondary_preds: Sequence[int],
    neutral_label: int = 1,
) -> list[int]:
    if len(primary_preds) != len(secondary_preds):
        raise ValueError("prediction sequences must have the same length")

    fused: list[int] = []
    for primary, secondary in zip(primary_preds, secondary_preds):
        if primary == secondary:
            fused.append(int(primary))
        elif neutral_label in (primary, secondary):
            fused.append(int(neutral_label))
        else:
            fused.append(int(primary))
    return fused


def build_pair_analysis(
    y_true: Sequence[int],
    primary_preds: Sequence[int],
    secondary_preds: Sequence[int],
    label_map: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    if label_map is None:
        label_map = LABEL_MAP

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[int]] = {}
    for truth, primary, secondary in zip(y_true, primary_preds, secondary_preds):
        grouped.setdefault((int(primary), int(secondary)), []).append(int(truth))

    for (primary, secondary), truths in sorted(grouped.items()):
        truth_counts = Counter(truths)
        total = len(truths)
        rows.append(
            {
                "primary_pred": primary,
                "secondary_pred": secondary,
                "primary_label": label_map[primary],
                "secondary_label": label_map[secondary],
                "count": total,
                "truth_mode": label_map[truth_counts.most_common(1)[0][0]],
                "truth_distribution": json.dumps(
                    {label_map[label]: count for label, count in sorted(truth_counts.items())},
                    ensure_ascii=False,
                ),
                "primary_correct_rate": truth_counts.get(primary, 0) / total,
                "secondary_correct_rate": truth_counts.get(secondary, 0) / total,
                "has_neutral_disagreement": int(primary != secondary and 1 in (primary, secondary)),
            }
        )

    return pd.DataFrame(rows)


def build_metric_row(
    model_name: str,
    split: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "Model": model_name,
        "Split": split,
        "Accuracy": metrics["accuracy"],
        "Precision_Macro": metrics["precision_macro"],
        "Recall_Macro": metrics["recall_macro"],
        "F1_Macro": metrics["f1_macro"],
        "F1_Weighted": metrics["f1_weighted"],
        "F1_Negative": metrics["f1_per_class"][0],
        "F1_Neutral": metrics["f1_per_class"][1],
        "F1_Positive": metrics["f1_per_class"][2],
    }
