from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    extract_swn_features_batch,
    extract_swn_features_extended_batch,
    load_data,
    load_sentiwordnet,
    preprocess_vietnamese,
)
from src.hybrid_fusion import search_weighted_feature_fusion, simplex_weight_grid


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    raise FileNotFoundError("Project root not found")


class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled

    def extract_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]


def load_model_safe(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(cleaned)
    return model


def extract_phobert_embeddings(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: PhoBERTClassifier,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    embeddings = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        batch_emb = model.extract_embeddings(inputs["input_ids"], inputs["attention_mask"])
        embeddings.append(batch_emb.cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)


def load_or_create_embeddings(
    split_name: str,
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: PhoBERTClassifier,
    device: torch.device,
    batch_size: int,
    max_length: int,
    cache_dir: Path,
) -> np.ndarray:
    cache_path = cache_dir / f"{split_name}_phobert.npy"
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = extract_phobert_embeddings(texts, tokenizer, model, device, batch_size, max_length)
    np.save(cache_path, embeddings)
    return embeddings


def parse_csv_list(raw: str, cast=float) -> list:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def logistic_param_grid(c_values: Iterable[float], class_weights: Iterable[str | None]) -> list[Dict[str, object]]:
    grid = []
    for c_value, class_weight in product(c_values, class_weights):
        grid.append(
            {
                "C": float(c_value),
                "class_weight": None if class_weight in (None, "none", "None") else class_weight,
            }
        )
    return grid


def train_logistic_regression(features: Dict[str, np.ndarray], params: Dict[str, object]) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
        n_jobs=-1,
        C=float(params["C"]),
        class_weight=params["class_weight"],
    )
    model.fit(features["train"], TRAIN_LABELS)
    return model


@dataclass
class RunConfig:
    project_root: Path
    results_dir: Path
    data_dir: Path
    sentiwordnet_path: Path
    phobert_model_path: Path
    model_name: str
    mode: str
    use_extended_swn: bool
    tfidf_max_features: int
    tfidf_min_df: int
    tfidf_max_df: float
    tfidf_ngram_max: int
    batch_size: int
    max_length: int
    weight_step: float


TRAIN_LABELS: list[int] = []


def build_feature_blocks(config: RunConfig) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Sequence[int]]]:
    train_texts, train_labels = load_data(str(config.data_dir), "train")
    val_texts, val_labels = load_data(str(config.data_dir), "validation")
    test_texts, test_labels = load_data(str(config.data_dir), "test")

    labels = {
        "train": train_labels,
        "val": val_labels,
        "test": test_labels,
    }

    global TRAIN_LABELS
    TRAIN_LABELS = train_labels

    cache_dir = config.results_dir / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    phobert_model = PhoBERTClassifier(model_name=config.model_name, num_classes=3)
    phobert_model = load_model_safe(phobert_model, config.phobert_model_path, device)
    phobert_model = phobert_model.to(device)
    for param in phobert_model.parameters():
        param.requires_grad = False

    train_phobert = load_or_create_embeddings(
        "train",
        train_texts,
        tokenizer,
        phobert_model,
        device,
        config.batch_size,
        config.max_length,
        cache_dir,
    )
    val_phobert = load_or_create_embeddings(
        "validation",
        val_texts,
        tokenizer,
        phobert_model,
        device,
        config.batch_size,
        config.max_length,
        cache_dir,
    )
    test_phobert = load_or_create_embeddings(
        "test",
        test_texts,
        tokenizer,
        phobert_model,
        device,
        config.batch_size,
        config.max_length,
        cache_dir,
    )

    phobert_scaler = StandardScaler()
    feature_blocks: Dict[str, Dict[str, np.ndarray]] = {
        "phobert": {
            "train": phobert_scaler.fit_transform(train_phobert).astype(np.float32),
            "val": phobert_scaler.transform(val_phobert).astype(np.float32),
            "test": phobert_scaler.transform(test_phobert).astype(np.float32),
        }
    }
    joblib.dump(phobert_scaler, cache_dir / "phobert_scaler.pkl")

    if "tfidf" in config.mode:
        train_processed = [preprocess_vietnamese(text) for text in train_texts]
        val_processed = [preprocess_vietnamese(text) for text in val_texts]
        test_processed = [preprocess_vietnamese(text) for text in test_texts]

        tfidf_vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=(1, config.tfidf_ngram_max),
            min_df=config.tfidf_min_df,
            max_df=config.tfidf_max_df,
            sublinear_tf=True,
        )
        feature_blocks["tfidf"] = {
            "train": tfidf_vectorizer.fit_transform(train_processed).astype(np.float32).toarray(),
            "val": tfidf_vectorizer.transform(val_processed).astype(np.float32).toarray(),
            "test": tfidf_vectorizer.transform(test_processed).astype(np.float32).toarray(),
        }
        joblib.dump(tfidf_vectorizer, cache_dir / "tfidf_vectorizer.pkl")

    if "swn" in config.mode:
        word_to_scores = load_sentiwordnet(str(config.sentiwordnet_path))
        extract_batch = extract_swn_features_extended_batch if config.use_extended_swn else extract_swn_features_batch

        train_swn = extract_batch(train_texts, word_to_scores).astype(np.float32)
        val_swn = extract_batch(val_texts, word_to_scores).astype(np.float32)
        test_swn = extract_batch(test_texts, word_to_scores).astype(np.float32)

        swn_scaler = StandardScaler()
        feature_blocks["swn"] = {
            "train": swn_scaler.fit_transform(train_swn).astype(np.float32),
            "val": swn_scaler.transform(val_swn).astype(np.float32),
            "test": swn_scaler.transform(test_swn).astype(np.float32),
        }
        joblib.dump(swn_scaler, cache_dir / "swn_scaler.pkl")

    return feature_blocks, labels


def write_outputs(config: RunConfig, best_run, candidate_count: int) -> None:
    summaries_dir = config.results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {
                "Split": "Train",
                **{f"Weight_{name}": value for name, value in best_run.weights.items()},
                **best_run.params,
                "Accuracy": best_run.train_metrics["accuracy"],
                "F1_Macro": best_run.train_metrics["f1_macro"],
                "F1_Weighted": best_run.train_metrics["f1_weighted"],
                "F1_Negative": best_run.train_metrics["f1_per_class"][0],
                "F1_Neutral": best_run.train_metrics["f1_per_class"][1],
                "F1_Positive": best_run.train_metrics["f1_per_class"][2],
            },
            {
                "Split": "Validation",
                **{f"Weight_{name}": value for name, value in best_run.weights.items()},
                **best_run.params,
                "Accuracy": best_run.val_metrics["accuracy"],
                "F1_Macro": best_run.val_metrics["f1_macro"],
                "F1_Weighted": best_run.val_metrics["f1_weighted"],
                "F1_Negative": best_run.val_metrics["f1_per_class"][0],
                "F1_Neutral": best_run.val_metrics["f1_per_class"][1],
                "F1_Positive": best_run.val_metrics["f1_per_class"][2],
            },
            {
                "Split": "Test",
                **{f"Weight_{name}": value for name, value in best_run.weights.items()},
                **best_run.params,
                "Accuracy": best_run.test_metrics["accuracy"],
                "F1_Macro": best_run.test_metrics["f1_macro"],
                "F1_Weighted": best_run.test_metrics["f1_weighted"],
                "F1_Negative": best_run.test_metrics["f1_per_class"][0],
                "F1_Neutral": best_run.test_metrics["f1_per_class"][1],
                "F1_Positive": best_run.test_metrics["f1_per_class"][2],
            },
        ]
    )
    summary_df.to_csv(summaries_dir / "summary.csv", index=False)

    experiment_summary = {
        "mode": config.mode,
        "timestamp": config.results_dir.name,
        "weight_step": config.weight_step,
        "candidate_count": candidate_count,
        "best_weights": best_run.weights,
        "best_params": best_run.params,
        "validation": best_run.val_metrics,
        "test": best_run.test_metrics,
    }
    with open(summaries_dir / "experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, ensure_ascii=False, indent=2)

    lines = [
        "=" * 60,
        "HYBRID WEIGHT TUNING RESULTS",
        "=" * 60,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {config.mode}",
        f"Results Dir: {config.results_dir}",
        f"Weight Step: {config.weight_step}",
        f"Candidates Evaluated: {candidate_count}",
        "",
        "-" * 60,
        "BEST WEIGHTS",
        "-" * 60,
    ]
    for name, value in best_run.weights.items():
        lines.append(f"{name}: {value:.4f}")
    lines.extend(
        [
            "",
            "-" * 60,
            "BEST CLASSIFIER PARAMS",
            "-" * 60,
        ]
    )
    for name, value in best_run.params.items():
        lines.append(f"{name}: {value}")
    lines.extend(
        [
            "",
            "-" * 60,
            "TEST RESULTS",
            "-" * 60,
            f"Accuracy: {best_run.test_metrics['accuracy']:.4f}",
            f"F1 Macro: {best_run.test_metrics['f1_macro']:.4f}",
            f"F1 Weighted: {best_run.test_metrics['f1_weighted']:.4f}",
            f"F1 Negative: {best_run.test_metrics['f1_per_class'][0]:.4f}",
            f"F1 Neutral: {best_run.test_metrics['f1_per_class'][1]:.4f}",
            f"F1 Positive: {best_run.test_metrics['f1_per_class'][2]:.4f}",
        ]
    )
    with open(summaries_dir / "training_results.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune PhoBERT / TF-IDF / SentiWordNet feature weights.")
    parser.add_argument(
        "--mode",
        choices=["phobert_swn", "phobert_tfidf_swn"],
        default="phobert_tfidf_swn",
        help="Which hybrid feature stack to tune.",
    )
    parser.add_argument("--weight-step", type=float, default=None, help="Simplex step for weight search.")
    parser.add_argument("--c-values", default="0.01,0.1,1.0,5.0", help="Comma-separated C values.")
    parser.add_argument("--class-weights", default="none,balanced", help="Comma-separated class_weight values.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=3)
    parser.add_argument("--tfidf-max-df", type=float, default=0.9)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--use-basic-swn", action="store_true", help="Use 8 SentiWordNet features instead of 35.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    project_root = find_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_step = args.weight_step if args.weight_step is not None else (0.1 if args.mode == "phobert_tfidf_swn" else 0.05)

    config = RunConfig(
        project_root=project_root,
        results_dir=project_root / "results" / "Hybrid_Weight_Tuning" / args.mode / timestamp,
        data_dir=project_root / "data" / "processed",
        sentiwordnet_path=project_root / "data" / "sentiwordnet-dataset" / "VietSentiWordnet_Ver1.3.5.txt",
        phobert_model_path=project_root / "results" / "PhoBERT" / "baseline" / "models" / "phobert_model.pt",
        model_name="vinai/phobert-base",
        mode=args.mode,
        use_extended_swn=not args.use_basic_swn,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_max_df=args.tfidf_max_df,
        tfidf_ngram_max=args.tfidf_ngram_max,
        batch_size=args.batch_size,
        max_length=args.max_length,
        weight_step=weight_step,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "models").mkdir(parents=True, exist_ok=True)

    feature_blocks, labels = build_feature_blocks(config)
    block_names = list(feature_blocks.keys())
    weight_candidates = list(simplex_weight_grid(block_names, step=config.weight_step))
    param_grid = logistic_param_grid(
        parse_csv_list(args.c_values, float),
        parse_csv_list(args.class_weights, str),
    )

    best_run = search_weighted_feature_fusion(
        feature_blocks=feature_blocks,
        labels=labels,
        train_model=train_logistic_regression,
        weight_candidates=weight_candidates,
        param_grid=param_grid,
        selection_metric="f1_macro",
    )

    joblib.dump(best_run.model, config.results_dir / "models" / "best_model.pkl")
    write_outputs(config, best_run, candidate_count=len(weight_candidates) * len(param_grid))

    print(f"Mode: {config.mode}")
    print(f"Best weights: {best_run.weights}")
    print(f"Best params: {best_run.params}")
    print(f"Validation macro F1: {best_run.val_metrics['f1_macro']:.4f}")
    print(f"Test macro F1: {best_run.test_metrics['f1_macro']:.4f}")
    print(f"Saved to: {config.results_dir}")


if __name__ == "__main__":
    main()
