"""
Build Dataset - Tiền xử lý dữ liệu cho các models
==================================================

Script này sẽ:
1. Load dataset từ cache
2. Tiền xử lý text (clean, tokenize)
3. Tạo features cho các models (TF-IDF cho SVM, embeddings cho LSTM/PhoBERT)
4. Lưu dataset đã xử lý
"""

import os
import sys
from pathlib import Path
import pickle
import json

# Thêm thư mục gốc vào path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import yaml
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load config
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config():
    """Load cấu hình từ file yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text, config):
    """
    Làm sạch văn bản tiếng Việt
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase nếu được cấu hình
    if config.get("preprocessing", {}).get("lowercase", True):
        text = text.lower()
    
    # Remove special characters
    if config.get("preprocessing", {}).get("remove_special_chars", True):
        # Giữ lại chữ cái tiếng Việt và khoảng trắng
        text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', text)
    
    # Remove extra spaces
    if config.get("preprocessing", {}).get("remove_extra_spaces", True):
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_dataset(config):
    """Load dataset từ CSV files"""
    raw_data_dir = ROOT_DIR / config["paths"]["raw_data"]
    
    data = {}
    for split in ["train", "validation", "test"]:
        csv_path = raw_data_dir / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data[split] = df
            print(f"  ✓ Loaded {split}: {len(df)} samples")
        else:
            print(f"  ⚠️ {split}.csv not found")
    
    return data


def preprocess_dataset(data, config):
    """
    Tiền xử lý toàn bộ dataset
    """
    processed_data = {}
    
    for split, df in data.items():
        # Clean text
        df['clean_sentence'] = df['sentence'].apply(lambda x: clean_text(x, config))
        processed_data[split] = df
        print(f"  ✓ Preprocessed {split}: {len(df)} samples")
    
    return processed_data


def build_tfidf_features(processed_data, config):
    """
    Xây dựng TF-IDF features cho SVM
    """
    print("\n📊 Building TF-IDF features for SVM...")
    
    # Fit TF-IDF trên tập train
    train_texts = processed_data["train"]["clean_sentence"].tolist()
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit và transform
    X_train = tfidf.fit_transform(train_texts)
    print(f"  ✓ TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"  ✓ Train features shape: {X_train.shape}")
    
    # Transform validation và test
    features = {"train": X_train}
    
    if "validation" in processed_data:
        X_val = tfidf.transform(processed_data["validation"]["clean_sentence"].tolist())
        features["validation"] = X_val
        print(f"  ✓ Validation features shape: {X_val.shape}")
    
    if "test" in processed_data:
        X_test = tfidf.transform(processed_data["test"]["clean_sentence"].tolist())
        features["test"] = X_test
        print(f"  ✓ Test features shape: {X_test.shape}")
    
    return tfidf, features


def build_labels(processed_data):
    """
    Xây dựng labels cho các models
    """
    print("\n🏷️ Building labels...")
    
    labels = {}
    for split, df in processed_data.items():
        labels[split] = {
            "sentiment": df["sentiment"].values,
            "topic": df["topic"].values
        }
        print(f"  ✓ {split} labels: sentiment={len(df)}, topic={len(df)}")
    
    return labels


def save_processed_data(processed_data, tfidf, features, labels, config):
    """
    Lưu dữ liệu đã xử lý
    """
    processed_dir = ROOT_DIR / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving processed data to {processed_dir}...")
    
    # Lưu processed DataFrames
    for split, df in processed_data.items():
        df.to_csv(processed_dir / f"{split}_processed.csv", index=False)
        print(f"  ✓ Saved {split}_processed.csv")
    
    # Lưu TF-IDF vectorizer
    tfidf_path = processed_dir / "tfidf_vectorizer.pkl"
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"  ✓ Saved tfidf_vectorizer.pkl")
    
    # Lưu TF-IDF features
    for split, X in features.items():
        feature_path = processed_dir / f"tfidf_{split}.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump(X, f)
        print(f"  ✓ Saved tfidf_{split}.pkl")
    
    # Lưu labels
    labels_path = processed_dir / "labels.pkl"
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    print(f"  ✓ Saved labels.pkl")
    
    # Lưu metadata
    metadata = {
        "num_samples": {split: len(df) for split, df in processed_data.items()},
        "tfidf_vocab_size": len(tfidf.vocabulary_),
        "sentiment_classes": 3,
        "topic_classes": 4
    }
    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")


def main():
    """Main function"""
    print(f"\n{'#'*60}")
    print("# BUILD DATASET - Preprocessing Pipeline")
    print(f"{'#'*60}")
    
    # Load config
    print("\n📋 Loading configuration...")
    config = load_config()
    
    # Load dataset
    print("\n📥 Loading dataset...")
    data = load_dataset(config)
    
    if not data:
        print("❌ No dataset found! Run 'python data/download_dataset.py' first.")
        return
    
    # Preprocess
    print("\n🧹 Preprocessing text...")
    processed_data = preprocess_dataset(data, config)
    
    # Build TF-IDF features
    tfidf, features = build_tfidf_features(processed_data, config)
    
    # Build labels
    labels = build_labels(processed_data)
    
    # Save
    save_processed_data(processed_data, tfidf, features, labels, config)
    
    print(f"\n{'='*60}")
    print("✅ PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Train SVM: python training/train_svm.py")
    print("  2. Train LSTM: python training/train_lstm.py")
    print("  3. Train PhoBERT: python training/train_phobert.py")


if __name__ == "__main__":
    main()
