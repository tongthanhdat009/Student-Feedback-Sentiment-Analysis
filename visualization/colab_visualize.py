"""
Visualization cho Google Colab
==============================

Script này hiển thị các biểu đồ kết quả training bằng matplotlib.
Chạy trực tiếp trên Google Colab.

Cách sử dụng:
    # Trong Colab notebook
    %matplotlib inline
    from visualization.colab_visualize import *
    
    # Hiển thị tất cả biểu đồ
    show_all_results()
"""

import os
import sys
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Setup matplotlib cho Colab
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Thêm root directory vào path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Đường dẫn
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"


def load_results(model_name):
    """Load kết quả training của một model"""
    results_path = SAVED_MODELS_DIR / model_name / "results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return None


def plot_training_history(model_name="phobert", figsize=(14, 5)):
    """
    Vẽ biểu đồ training history (Loss và Accuracy)
    
    Args:
        model_name: "phobert", "lstm", hoặc "svm"
        figsize: Kích thước figure
    """
    results = load_results(model_name)
    
    if not results:
        print(f"❌ Không tìm thấy kết quả cho {model_name}")
        return
    
    train_history = results.get("train_history", [])
    val_history = results.get("val_history", [])
    
    if not train_history:
        print(f"❌ Không có training history cho {model_name}")
        return
    
    epochs = range(1, len(train_history) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot Loss
    ax1 = axes[0]
    train_loss = [h["loss"] for h in train_history]
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name.upper()} - Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2 = axes[1]
    train_acc = [h["accuracy"] for h in train_history]
    ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    
    if val_history:
        val_acc = [h["accuracy"] for h in val_history]
        ax2.plot(epochs, val_acc, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name.upper()} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
    
    # In kết quả cuối cùng
    print(f"\n📊 {model_name.upper()} Results:")
    if "test" in results:
        print(f"   Test Accuracy: {results['test']['accuracy']:.4f}")
        print(f"   Test F1-Score: {results['test']['f1']:.4f}")
    elif "validation" in results:
        print(f"   Val Accuracy: {results['validation']['accuracy']:.4f}")
        print(f"   Val F1-Score: {results['validation']['f1']:.4f}")


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", figsize=(8, 6)):
    """
    Vẽ confusion matrix
    
    Args:
        y_true: Labels thực tế
        y_pred: Labels dự đoán
        labels: Tên các class
        title: Tiêu đề
        figsize: Kích thước figure
    """
    if labels is None:
        labels = ["Negative", "Neutral", "Positive"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    # In thêm tỷ lệ
    print(f"\n📊 Confusion Matrix Analysis:")
    total = cm.sum()
    for i, label in enumerate(labels):
        correct = cm[i, i]
        total_class = cm[i, :].sum()
        print(f"   {label}: {correct}/{total_class} ({correct/total_class*100:.1f}%)")


def plot_model_comparison(figsize=(12, 5)):
    """
    So sánh kết quả giữa các models
    """
    models = ["svm", "lstm", "phobert"]
    results_data = []
    
    for model in models:
        results = load_results(model)
        if results:
            test_results = results.get("test", results.get("validation", {}))
            if test_results:
                results_data.append({
                    "Model": model.upper(),
                    "Accuracy": test_results.get("accuracy", 0),
                    "F1-Score": test_results.get("f1", 0)
                })
    
    if not results_data:
        print("❌ Không tìm thấy kết quả của bất kỳ model nào")
        return
    
    df = pd.DataFrame(results_data)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy comparison
    ax1 = axes[0]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars1 = ax1.bar(df["Model"], df["Accuracy"], color=colors[:len(df)])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Comparison - Accuracy')
    ax1.set_ylim([0, 1])
    
    # Thêm giá trị lên bar
    for bar, val in zip(bars1, df["Accuracy"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    ax2 = axes[1]
    bars2 = ax2.bar(df["Model"], df["F1-Score"], color=colors[:len(df)])
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Model Comparison - F1-Score')
    ax2.set_ylim([0, 1])
    
    for bar, val in zip(bars2, df["F1-Score"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # In bảng so sánh
    print("\n📊 Model Comparison Table:")
    print(df.to_string(index=False))


def plot_class_distribution(figsize=(10, 5)):
    """
    Vẽ biểu đồ phân bố các class trong dataset
    """
    processed_dir = ROOT_DIR / "data" / "processed" / "acsa"
    
    data = {}
    for split in ["train", "validation", "test"]:
        path = processed_dir / f"{split}_processed.csv"
        if path.exists():
            df = pd.read_csv(path)
            data[split] = df["sentiment"].value_counts().sort_index()
    
    if not data:
        print("❌ Không tìm thấy dữ liệu processed")
        return
    
    labels = ["Negative", "Neutral", "Positive"]
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, (split, counts) in enumerate(data.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, counts.values, width, label=split.capitalize(), color=colors[i])
        
        # Thêm giá trị
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   str(val), ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_metrics_radar(figsize=(8, 8)):
    """
    Vẽ radar chart so sánh metrics của các models
    """
    models = ["svm", "lstm", "phobert"]
    metrics = ["Accuracy", "F1-Score"]
    
    data = {}
    for model in models:
        results = load_results(model)
        if results:
            test_results = results.get("test", results.get("validation", {}))
            if test_results:
                data[model.upper()] = [
                    test_results.get("accuracy", 0),
                    test_results.get("f1", 0)
                ]
    
    if not data:
        print("❌ Không tìm thấy kết quả")
        return
    
    # Tạo radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng tròn
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, (model, values) in enumerate(data.items()):
        values = values + values[:1]  # Đóng vòng tròn
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Comparison', pad=20)
    
    plt.tight_layout()
    plt.show()


def plot_learning_curves_all(figsize=(15, 10)):
    """
    Vẽ learning curves của tất cả models trong cùng một figure
    """
    models = ["lstm", "phobert"]  # SVM không có learning curve
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = {'lstm': '#2ecc71', 'phobert': '#e74c3c'}
    
    for model in models:
        results = load_results(model)
        if not results:
            continue
        
        train_history = results.get("train_history", [])
        val_history = results.get("val_history", [])
        
        if not train_history:
            continue
        
        epochs = range(1, len(train_history) + 1)
        color = colors[model]
        
        # Loss
        ax_loss = axes[0, 0] if model == "lstm" else axes[0, 1]
        train_loss = [h["loss"] for h in train_history]
        ax_loss.plot(epochs, train_loss, '-o', color=color, linewidth=2, markersize=4)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{model.upper()} - Training Loss')
        ax_loss.grid(True, alpha=0.3)
        
        # Accuracy
        ax_acc = axes[1, 0] if model == "lstm" else axes[1, 1]
        train_acc = [h["accuracy"] for h in train_history]
        ax_acc.plot(epochs, train_acc, '-o', color=color, linewidth=2, markersize=4, label='Train')
        
        if val_history:
            val_acc = [h["accuracy"] for h in val_history]
            ax_acc.plot(epochs, val_acc, '--s', color=color, linewidth=2, markersize=4, 
                       alpha=0.7, label='Validation')
        
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'{model.upper()} - Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def show_classification_report_visual(y_true, y_pred, labels=None, figsize=(10, 6)):
    """
    Hiển thị classification report dưới dạng heatmap
    """
    if labels is None:
        labels = ["Negative", "Neutral", "Positive"]
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Tạo DataFrame từ report
    df = pd.DataFrame(report).T
    df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df = df[['precision', 'recall', 'f1-score']]
    
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt='.2f')
    plt.title('Classification Report Heatmap')
    plt.tight_layout()
    plt.show()


def show_all_results():
    """
    Hiển thị tất cả các biểu đồ
    """
    print("=" * 60)
    print("📊 VISUALIZATION KẾT QUẢ TRAINING")
    print("=" * 60)
    
    # 1. Class distribution
    print("\n📈 1. Phân bố dữ liệu")
    print("-" * 40)
    try:
        plot_class_distribution()
    except Exception as e:
        print(f"   ⚠️ Không thể vẽ: {e}")
    
    # 2. Model comparison
    print("\n📈 2. So sánh các Models")
    print("-" * 40)
    try:
        plot_model_comparison()
    except Exception as e:
        print(f"   ⚠️ Không thể vẽ: {e}")
    
    # 3. Training histories
    print("\n📈 3. Training History - LSTM")
    print("-" * 40)
    try:
        plot_training_history("lstm")
    except Exception as e:
        print(f"   ⚠️ Không thể vẽ: {e}")
    
    print("\n📈 4. Training History - PhoBERT")
    print("-" * 40)
    try:
        plot_training_history("phobert")
    except Exception as e:
        print(f"   ⚠️ Không thể vẽ: {e}")
    
    # 4. All learning curves
    print("\n📈 5. Tất cả Learning Curves")
    print("-" * 40)
    try:
        plot_learning_curves_all()
    except Exception as e:
        print(f"   ⚠️ Không thể vẽ: {e}")
    
    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT VISUALIZATION!")
    print("=" * 60)


# ============================================================
# HƯỚNG DẪN SỬ DỤNG TRONG GOOGLE COLAB
# ============================================================
"""
📋 HƯỚNG DẪN SỬ DỤNG TRONG GOOGLE COLAB:

# Cell 1: Setup
%matplotlib inline
import sys
sys.path.insert(0, '/content/drive/MyDrive/Student-Feedback-Sentiment-Analysis')

# Cell 2: Import
from visualization.colab_visualize import *

# Cell 3: Hiển thị tất cả biểu đồ
show_all_results()

# Hoặc hiển thị từng biểu đồ riêng lẻ:
plot_class_distribution()           # Phân bố dữ liệu
plot_model_comparison()             # So sánh models
plot_training_history("lstm")       # Learning curve LSTM
plot_training_history("phobert")    # Learning curve PhoBERT
plot_learning_curves_all()          # Tất cả learning curves
"""

if __name__ == "__main__":
    print("\n📊 Colab Visualization Module")
    print("=" * 40)
    print("Sử dụng trong Colab:")
    print("  from visualization.colab_visualize import *")
    print("  show_all_results()")
    print()
    
    # Thử hiển thị nếu chạy trực tiếp
    try:
        show_all_results()
    except Exception as e:
        print(f"⚠️ Lỗi: {e}")
        print("   Có thể chưa có kết quả training.")
