"""
Train LSTM Model - Long Short-Term Memory
==========================================

Script này sẽ:
1. Load preprocessed data
2. Tạo word embeddings
3. Train LSTM classifier
4. Evaluate và lưu model
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
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
from tqdm import tqdm

# Load config
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Vocabulary:
    """Build vocabulary từ text"""
    def __init__(self, min_freq=2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq
    
    def build(self, texts):
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        idx = 2
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        return self
    
    def encode(self, text, max_len=128):
        words = text.split()[:max_len]
        indices = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>
        return indices
    
    def __len__(self):
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """Dataset cho LSTM"""
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        indices = self.vocab.encode(text, self.max_len)
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences to same length"""
    texts, labels = zip(*batch)
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels


class LSTMClassifier(nn.Module):
    """LSTM model for sentiment classification"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Dropout and FC
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output


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


def train_epoch(model, dataloader, criterion, optimizer):
    """Train một epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for texts, labels in dataloader:
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def evaluate(model, dataloader):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    return acc, f1, all_preds, all_labels


def save_model(model, vocab, results, config):
    """Lưu model và kết quả"""
    save_dir = ROOT_DIR / config["paths"]["saved_models"] / "lstm"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to {save_dir}...")
    
    # Lưu model
    torch.save(model.state_dict(), save_dir / "lstm_sentiment.pt")
    print(f"  ✓ Saved lstm_sentiment.pt")
    
    # Lưu vocab
    with open(save_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"  ✓ Saved vocab.pkl")
    
    # Lưu results
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results.json")


def main():
    """Main function"""
    print(f"\n{'#'*60}")
    print("# TRAIN LSTM - Long Short-Term Memory")
    print(f"{'#'*60}")
    print(f"\n🖥️ Device: {device}")
    
    # Load config
    config = load_config()
    lstm_config = config.get("models", {}).get("lstm", {})
    
    # Load data
    data = load_processed_data(config)
    
    if "train" not in data:
        print("❌ No training data! Run preprocessing first.")
        return
    
    # Build vocabulary
    print("\n📚 Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build(data["train"]["clean_sentence"].tolist())
    print(f"  ✓ Vocabulary size: {len(vocab)}")
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = SentimentDataset(
        data["train"]["clean_sentence"].tolist(),
        data["train"]["sentiment"].tolist(),
        vocab
    )
    
    val_dataset = None
    if "validation" in data:
        val_dataset = SentimentDataset(
            data["validation"]["clean_sentence"].tolist(),
            data["validation"]["sentiment"].tolist(),
            vocab
        )
    
    test_dataset = None
    if "test" in data:
        test_dataset = SentimentDataset(
            data["test"]["clean_sentence"].tolist(),
            data["test"]["sentiment"].tolist(),
            vocab
        )
    
    # Create dataloaders
    batch_size = lstm_config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn) if test_dataset else None
    
    # Create model
    print("\n🤖 Creating LSTM model...")
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=lstm_config.get("embedding_dim", 256),
        hidden_dim=lstm_config.get("hidden_dim", 128),
        num_layers=lstm_config.get("num_layers", 2),
        num_classes=3,
        dropout=lstm_config.get("dropout", 0.3)
    ).to(device)
    
    print(f"  Embedding dim: {lstm_config.get('embedding_dim', 256)}")
    print(f"  Hidden dim: {lstm_config.get('hidden_dim', 128)}")
    print(f"  Num layers: {lstm_config.get('num_layers', 2)}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_config.get("learning_rate", 0.001))
    epochs = lstm_config.get("epochs", 20)
    
    # Train
    print(f"\n🏋️ Training for {epochs} epochs...")
    results = {"train_history": [], "val_history": []}
    best_val_acc = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        results["train_history"].append({"loss": train_loss, "accuracy": train_acc})
        
        log = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        
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
    save_model(model, vocab, results, config)
    
    print(f"\n{'='*60}")
    print("✅ LSTM TRAINING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
