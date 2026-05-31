import csv
import json
from pathlib import Path
from typing import Any


CANDIDATE_JSON = ("training_results.json", "results.json", "metrics.json")


def _scalar(value: str) -> Any:
    text = value.strip()
    if not text: return text
    lowered = text.lower()
    if lowered in {"true", "false"}: return lowered == "true"
    try:
        if any(ch in text for ch in [".", "e", "E"]): return float(text)
        return int(text)
    except ValueError:
        return text


def _bucket(key: str) -> str:
    k = key.lower()
    if any(x in k for x in ["accuracy", "acc", "f1", "precision", "recall", "loss", "auc"]): return "metrics"
    return "params"


def _normalize(source_file: str, data: Any, warnings: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"source_file": source_file, "metrics": {}, "params": {}, "artifacts": [], "warnings": warnings}
    if isinstance(data, dict):
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        result["metrics"].update(metrics)
        result["params"].update(params)
        for key, value in data.items():
            if key in {"metrics", "params", "artifacts", "warnings"}: continue
            result[_bucket(key)][key] = value
        if isinstance(data.get("artifacts"), list): result["artifacts"] = data["artifacts"]
        if isinstance(data.get("warnings"), list): result["warnings"].extend(str(x) for x in data["warnings"])
    return result


def parse_training_results(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    warnings: list[str] = []
    if not root.exists():
        return {"source_file": None, "metrics": {}, "params": {}, "artifacts": [], "warnings": ["output directory missing"]}

    for name in CANDIDATE_JSON:
        path = next((p for p in root.rglob(name) if p.is_file()), None)
        if not path: continue
        try:
            return _normalize(path.relative_to(root).as_posix(), json.loads(path.read_text(encoding="utf-8")), warnings)
        except Exception as exc:
            warnings.append(f"Failed to parse {name}: {exc}")

    txt = next((p for p in root.rglob("training_results.txt") if p.is_file()), None)
    if txt:
        data: dict[str, Any] = {}
        try:
            for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                sep = ":" if ":" in line else "=" if "=" in line else None
                if not sep: continue
                key, value = line.split(sep, 1)
                key = key.strip()
                if key: data[key] = _scalar(value)
            return _normalize(txt.relative_to(root).as_posix(), data, warnings)
        except Exception as exc:
            warnings.append(f"Failed to parse training_results.txt: {exc}")

    csv_file = next((p for p in root.rglob("*.csv") if p.is_file() and any(x in p.name.lower() for x in ["result", "metric", "summary"])), None)
    if csv_file:
        try:
            with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                row = next(csv.DictReader(fh), None) or {}
            return _normalize(csv_file.relative_to(root).as_posix(), {k: _scalar(v) for k, v in row.items() if k}, warnings)
        except Exception as exc:
            warnings.append(f"Failed to parse {csv_file.name}: {exc}")

    return {"source_file": None, "metrics": {}, "params": {}, "artifacts": [], "warnings": warnings or ["No training result file found"]}
