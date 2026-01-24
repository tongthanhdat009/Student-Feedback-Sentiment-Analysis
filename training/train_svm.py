"""
Train SVM Model - Support Vector Machine
=========================================

Script này sẽ:
1. Load preprocessed data
2. Train SVM classifier
3. Evaluate và lưu model
"""

import os
import sys
from pathlib import Path
import pickle
import json

# Thêm thư mục gốc vào path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import yaml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import report generator
from utils.report_generator import generate_training_report

# Load config
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_data(config):
    """Load dữ liệu đã tiền xử lý"""
    processed_dir = ROOT_DIR / config["paths"]["processed_data"]
    
    print("📥 Loading processed data...")
    
    # Load TF-IDF features
    features = {}
    for split in ["train", "validation", "test"]:
        path = processed_dir / f"tfidf_{split}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                features[split] = pickle.load(f)
            print(f"  ✓ Loaded {split} features: {features[split].shape}")
    
    # Load labels
    with open(processed_dir / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    print(f"  ✓ Loaded labels")
    
    return features, labels


def train_svm(X_train, y_train, config):
    """Train SVM model"""
    svm_config = config.get("models", {}).get("svm", {})
    
    print("\n🤖 Training SVM model...")
    print(f"  Kernel: {svm_config.get('kernel', 'rbf')}")
    print(f"  C: {svm_config.get('C', 1.0)}")
    
    model = SVC(
        kernel=svm_config.get("kernel", "rbf"),
        C=svm_config.get("C", 1.0),
        gamma=svm_config.get("gamma", "scale"),
        random_state=42,
        probability=True
    )
    
    model.fit(X_train, y_train)
    print("  ✓ Training complete!")
    
    return model


def evaluate_model(model, X, y, split_name):
    """Đánh giá model"""
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    
    print(f"\n📊 {split_name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Detailed report
    print(f"\n  Classification Report:")
    report = classification_report(y, y_pred, target_names=["Negative", "Neutral", "Positive"])
    for line in report.split('\n'):
        print(f"    {line}")
    
    return {"accuracy": acc, "f1": f1, "predictions": y_pred.tolist()}


def save_model(model, results, config):
    """Lưu model và kết quả"""
    save_dir = ROOT_DIR / config["paths"]["saved_models"] / "svm"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to {save_dir}...")
    
    # Lưu model
    model_path = save_dir / "svm_sentiment.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved svm_sentiment.pkl")
    
    # Lưu results
    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON
        json_results = {}
        for key, val in results.items():
            if isinstance(val, dict):
                json_results[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                     for k, v in val.items()}
            else:
                json_results[key] = val
        json.dump(json_results, f, indent=2)
    print(f"  ✓ Saved results.json")


def main():
    """Main function"""
    print(f"\n{'#'*60}")
    print("# TRAIN SVM - Support Vector Machine")
    print(f"{'#'*60}")
    
    # Load config
    config = load_config()
    
    # Load data
    features, labels = load_processed_data(config)
    
    if "train" not in features:
        print("❌ No training data! Run preprocessing first.")
        return
    
    X_train = features["train"]
    y_train = labels["train"]["sentiment"]
    
    # Train
    model = train_svm(X_train, y_train, config)
    
    # Evaluate
    results = {}
    
    # Train accuracy
    train_results = evaluate_model(model, X_train, y_train, "Train")
    results["train"] = train_results
    
    # Validation
    if "validation" in features:
        val_results = evaluate_model(
            model, features["validation"], 
            labels["validation"]["sentiment"], "Validation"
        )
        results["validation"] = val_results
    
    # Test
    if "test" in features:
        test_results = evaluate_model(
            model, features["test"], 
            labels["test"]["sentiment"], "Test"
        )
        results["test"] = test_results
    
    # Save
    save_model(model, results, config)
    
    # Lấy config cho SVM
    svm_config = config.get("models", {}).get("svm", {})
    
    # Tạo báo cáo và lưu vào file text
    try:
        generate_training_report("svm", results, svm_config)
    except Exception as e:
        print(f"⚠️ Không thể tạo report: {e}")
    
    print(f"\n{'='*60}")
    print("✅ SVM TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
