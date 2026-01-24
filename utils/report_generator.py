"""
Report Generator - Lưu kết quả training vào file text
======================================================

Module này tạo báo cáo chi tiết về kết quả training và lưu vào file text.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Root directory
ROOT_DIR = Path(__file__).parent.parent


def generate_training_report(model_name, results, config=None, save_path=None):
    """
    Tạo báo cáo training và lưu vào file text.
    
    Args:
        model_name: Tên model (svm, lstm, phobert)
        results: Dict chứa kết quả (train_history, val_history, test, validation)
        config: Config của model (optional)
        save_path: Đường dẫn lưu file (optional, mặc định là results/)
    
    Returns:
        Đường dẫn file report đã lưu
    """
    # Tạo nội dung report
    report_lines = []
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines.append("=" * 70)
    report_lines.append(f"  TRAINING REPORT - {model_name.upper()}")
    report_lines.append(f"  Generated: {timestamp}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Model info
    report_lines.append("-" * 70)
    report_lines.append("MODEL INFORMATION")
    report_lines.append("-" * 70)
    report_lines.append(f"Model Name: {model_name.upper()}")
    
    if config:
        report_lines.append(f"Config:")
        for key, value in config.items():
            report_lines.append(f"  - {key}: {value}")
    report_lines.append("")
    
    # Training History
    if "train_history" in results and results["train_history"]:
        report_lines.append("-" * 70)
        report_lines.append("TRAINING HISTORY")
        report_lines.append("-" * 70)
        
        train_history = results["train_history"]
        val_history = results.get("val_history", [])
        
        report_lines.append(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Val Acc':<15} {'Val F1':<15}")
        report_lines.append("-" * 70)
        
        for i, train_h in enumerate(train_history):
            epoch = i + 1
            train_loss = train_h.get("loss", 0)
            train_acc = train_h.get("accuracy", 0)
            
            val_acc = ""
            val_f1 = ""
            if i < len(val_history):
                val_acc = f"{val_history[i].get('accuracy', 0):.4f}"
                val_f1 = f"{val_history[i].get('f1', 0):.4f}"
            
            report_lines.append(f"{epoch:<8} {train_loss:<15.4f} {train_acc:<15.4f} {val_acc:<15} {val_f1:<15}")
        
        report_lines.append("")
    
    # Validation Results
    if "validation" in results:
        report_lines.append("-" * 70)
        report_lines.append("VALIDATION RESULTS")
        report_lines.append("-" * 70)
        val_results = results["validation"]
        report_lines.append(f"Accuracy: {val_results.get('accuracy', 0):.4f}")
        report_lines.append(f"F1-Score: {val_results.get('f1', 0):.4f}")
        report_lines.append("")
    
    # Test Results
    if "test" in results:
        report_lines.append("-" * 70)
        report_lines.append("TEST RESULTS")
        report_lines.append("-" * 70)
        test_results = results["test"]
        report_lines.append(f"Accuracy: {test_results.get('accuracy', 0):.4f}")
        report_lines.append(f"F1-Score: {test_results.get('f1', 0):.4f}")
        report_lines.append("")
    
    # Classification Report (nếu có)
    if "classification_report" in results:
        report_lines.append("-" * 70)
        report_lines.append("CLASSIFICATION REPORT")
        report_lines.append("-" * 70)
        report_lines.append(results["classification_report"])
        report_lines.append("")
    
    # Summary
    report_lines.append("-" * 70)
    report_lines.append("SUMMARY")
    report_lines.append("-" * 70)
    
    best_acc = 0
    best_f1 = 0
    
    if "test" in results:
        best_acc = results["test"].get("accuracy", 0)
        best_f1 = results["test"].get("f1", 0)
    elif "validation" in results:
        best_acc = results["validation"].get("accuracy", 0)
        best_f1 = results["validation"].get("f1", 0)
    
    report_lines.append(f"Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    report_lines.append(f"Best F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    report_lines.append("")
    
    # Footer
    report_lines.append("=" * 70)
    report_lines.append("  END OF REPORT")
    report_lines.append("=" * 70)
    
    # Tạo nội dung text
    report_content = "\n".join(report_lines)
    
    # Xác định đường dẫn lưu
    if save_path is None:
        results_dir = ROOT_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo tên file với timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_report_{timestamp_str}.txt"
        save_path = results_dir / filename
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"📄 Report saved to: {save_path}")
    
    return str(save_path)


def generate_summary_report(save_path=None):
    """
    Tạo báo cáo tổng hợp so sánh tất cả các models.
    
    Args:
        save_path: Đường dẫn lưu file (optional)
    
    Returns:
        Đường dẫn file report đã lưu
    """
    models = ["svm", "lstm", "phobert"]
    saved_models_dir = ROOT_DIR / "saved_models"
    
    report_lines = []
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines.append("=" * 70)
    report_lines.append("  SUMMARY REPORT - ALL MODELS COMPARISON")
    report_lines.append(f"  Generated: {timestamp}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Load results từ tất cả models
    all_results = {}
    for model in models:
        results_path = saved_models_dir / model / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                all_results[model] = json.load(f)
    
    if not all_results:
        report_lines.append("Không tìm thấy kết quả của bất kỳ model nào.")
        report_lines.append("Hãy train các models trước khi tạo summary report.")
    else:
        # Bảng so sánh
        report_lines.append("-" * 70)
        report_lines.append("MODEL COMPARISON")
        report_lines.append("-" * 70)
        report_lines.append(f"{'Model':<15} {'Test Accuracy':<20} {'Test F1-Score':<20}")
        report_lines.append("-" * 70)
        
        for model, results in all_results.items():
            test_results = results.get("test", results.get("validation", {}))
            acc = test_results.get("accuracy", 0)
            f1 = test_results.get("f1", 0)
            report_lines.append(f"{model.upper():<15} {acc:.4f} ({acc*100:.1f}%){'':<5} {f1:.4f} ({f1*100:.1f}%)")
        
        report_lines.append("")
        
        # Tìm model tốt nhất
        report_lines.append("-" * 70)
        report_lines.append("BEST MODEL")
        report_lines.append("-" * 70)
        
        best_model = None
        best_acc = 0
        for model, results in all_results.items():
            test_results = results.get("test", results.get("validation", {}))
            acc = test_results.get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_model = model
        
        if best_model:
            best_results = all_results[best_model]
            test_results = best_results.get("test", best_results.get("validation", {}))
            report_lines.append(f"Winner: {best_model.upper()}")
            report_lines.append(f"Accuracy: {test_results.get('accuracy', 0):.4f}")
            report_lines.append(f"F1-Score: {test_results.get('f1', 0):.4f}")
        
        report_lines.append("")
        
        # Chi tiết từng model
        for model, results in all_results.items():
            report_lines.append("-" * 70)
            report_lines.append(f"{model.upper()} DETAILS")
            report_lines.append("-" * 70)
            
            # Training history summary
            if "train_history" in results:
                train_history = results["train_history"]
                if train_history:
                    report_lines.append(f"Total Epochs: {len(train_history)}")
                    report_lines.append(f"Final Train Loss: {train_history[-1].get('loss', 0):.4f}")
                    report_lines.append(f"Final Train Acc: {train_history[-1].get('accuracy', 0):.4f}")
            
            # Test results
            if "test" in results:
                test_r = results["test"]
                report_lines.append(f"Test Accuracy: {test_r.get('accuracy', 0):.4f}")
                report_lines.append(f"Test F1-Score: {test_r.get('f1', 0):.4f}")
            
            report_lines.append("")
    
    # Footer
    report_lines.append("=" * 70)
    report_lines.append("  END OF SUMMARY REPORT")
    report_lines.append("=" * 70)
    
    # Tạo nội dung text
    report_content = "\n".join(report_lines)
    
    # Xác định đường dẫn lưu
    if save_path is None:
        results_dir = ROOT_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_report_{timestamp_str}.txt"
        save_path = results_dir / filename
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"📄 Summary report saved to: {save_path}")
    
    return str(save_path)


def print_report_to_console(model_name, results):
    """In report ra console"""
    print("\n" + "=" * 60)
    print(f"  {model_name.upper()} RESULTS")
    print("=" * 60)
    
    if "test" in results:
        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {results['test'].get('accuracy', 0):.4f}")
        print(f"   F1-Score: {results['test'].get('f1', 0):.4f}")
    
    if "validation" in results:
        print(f"\n📊 Validation Results:")
        print(f"   Accuracy: {results['validation'].get('accuracy', 0):.4f}")
        print(f"   F1-Score: {results['validation'].get('f1', 0):.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("📄 Report Generator Module")
    print("=" * 40)
    print("\nUsage:")
    print("  from utils.report_generator import generate_training_report, generate_summary_report")
    print("  generate_summary_report()  # Tạo báo cáo tổng hợp")
    print()
    
    # Thử tạo summary report nếu có dữ liệu
    try:
        generate_summary_report()
    except Exception as e:
        print(f"⚠️ Không thể tạo summary report: {e}")
