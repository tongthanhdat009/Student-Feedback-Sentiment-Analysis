"""
Train PhoBERT Model - Vietnamese BERT
=====================================

Script này sẽ:
1. Load preprocessed data
2. Fine-tune PhoBERT cho sentiment classification
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
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Import từ utils để sử dụng cache
from utils.cache_manager import load_phobert_model, get_model_cache_dir
from utils.report_generator import generate_training_report

# Load config
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PhoBERTDataset(Dataset):
    """Dataset cho PhoBERT"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_processed_data(config):
    """Load dữ liệu đã tiền xử lý"""
    processed_dir = ROOT_DIR / config["paths"]["processed_data"]
    
    print("📥 Loading processed data...")
    
    data = {}
    for split in ["train", "validation", "test"]:
        path = processed_dir / f"{split}_processed.csv"
        if path.exists():
            df = pd.read_csv(path)
            data[split] = df
            print(f"  ✓ Loaded {split}: {len(df)} samples")
    
    return data


def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    """Train một epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress.set_postfix({"loss": loss.item()})
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def evaluate(model, dataloader):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    return acc, f1, all_preds, all_labels


def save_model(model, tokenizer, results, config):
    """Lưu model và kết quả"""
    save_dir = ROOT_DIR / config["paths"]["saved_models"] / "phobert"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to {save_dir}...")
    
    # Lưu model
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  ✓ Saved model and tokenizer")
    
    # Lưu results
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results.json")


def main():
    """Main function"""
    print(f"\n{'#'*60}")
    print("# TRAIN PhoBERT - Vietnamese BERT")
    print(f"{'#'*60}")
    print(f"\n🖥️ Device: {device}")
    
    # Load config
    config = load_config()
    phobert_config = config.get("models", {}).get("phobert", {})
    
    # Load data
    data = load_processed_data(config)
    
    if "train" not in data:
        print("❌ No training data! Run preprocessing first.")
        return
    
    # Load PhoBERT từ cache
    print("\n🤖 Loading PhoBERT model (from cache if available)...")
    tokenizer, model = load_phobert_model(for_classification=True, num_labels=3, config=config)
    model = model.to(device)
    
    # Create datasets
    print("\n📦 Creating datasets...")
    max_length = phobert_config.get("max_length", 256)
    
    train_dataset = PhoBERTDataset(
        data["train"]["clean_sentence"].tolist(),
        data["train"]["sentiment"].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = None
    if "validation" in data:
        val_dataset = PhoBERTDataset(
            data["validation"]["clean_sentence"].tolist(),
            data["validation"]["sentiment"].tolist(),
            tokenizer,
            max_length
        )
    
    test_dataset = None
    if "test" in data:
        test_dataset = PhoBERTDataset(
            data["test"]["clean_sentence"].tolist(),
            data["test"]["sentiment"].tolist(),
            tokenizer,
            max_length
        )
    
    # Create dataloaders
    batch_size = phobert_config.get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    lr = phobert_config.get("learning_rate", 2e-5)
    # Đảm bảo learning rate là float (YAML có thể load 2e-5 như string)
    if isinstance(lr, str):
        lr = float(lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epochs = phobert_config.get("epochs", 5)
    
    # Train
    print(f"\n🏋️ Training for {epochs} epochs...")
    results = {"train_history": [], "val_history": []}
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\n📌 Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        results["train_history"].append({"loss": train_loss, "accuracy": train_acc})
        
        log = f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        
        if val_loader:
            val_acc, val_f1, _, _ = evaluate(model, val_loader)
            results["val_history"].append({"accuracy": val_acc, "f1": val_f1})
            log += f" | Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        print(f"  {log}")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    
    if val_loader:
        val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader)
        print(f"\n  Validation - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        results["validation"] = {"accuracy": val_acc, "f1": val_f1}
    
    if test_loader:
        test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader)
        print(f"  Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        results["test"] = {"accuracy": test_acc, "f1": test_f1}
        
        print("\n  Classification Report (Test):")
        report = classification_report(test_labels, test_preds, target_names=["Negative", "Neutral", "Positive"])
        for line in report.split('\n'):
            print(f"    {line}")
    
    # Save
    save_model(model, tokenizer, results, config)
    
    # Lưu classification report vào results
    if test_loader:
        report = classification_report(test_labels, test_preds, target_names=["Negative", "Neutral", "Positive"])
        results["classification_report"] = report
    
    # Tạo báo cáo và lưu vào file text
    try:
        generate_training_report("phobert", results, phobert_config)
    except Exception as e:
        print(f"⚠️ Không thể tạo report: {e}")
    
    print(f"\n{'='*60}")
    print("✅ PhoBERT TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
