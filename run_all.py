"""
RUN ALL - Student Feedback Sentiment Analysis
==============================================

Script Python chạy tất cả các bước: preprocessing, training, dashboard
Có thể chạy bằng: python run_all.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Thư mục gốc
ROOT_DIR = Path(__file__).parent

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


def print_step(step, total, name):
    print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.GREEN}[{step}/{total}] {name}{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}\n")


def run_script(script_path, description):
    """Chạy một Python script và kiểm tra kết quả"""
    print_step(run_script.step, 5, description)
    run_script.step += 1
    
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=str(ROOT_DIR)
    )
    
    if result.returncode != 0:
        print(f"\n{Colors.FAIL}ERROR: {description} failed!{Colors.ENDC}")
        return False
    
    return True

run_script.step = 1


def main():
    print(f"""
{Colors.CYAN}############################################################{Colors.ENDC}
{Colors.CYAN}# STUDENT FEEDBACK SENTIMENT ANALYSIS{Colors.ENDC}
{Colors.CYAN}# Running Full Pipeline{Colors.ENDC}
{Colors.CYAN}############################################################{Colors.ENDC}
    """)
    
    print(f"Working directory: {ROOT_DIR}")
    os.chdir(ROOT_DIR)
    
    # Step 1: Preprocessing
    if not run_script("preprocessing/build_dataset.py", "PREPROCESSING"):
        sys.exit(1)
    
    # Step 2: Train SVM
    if not run_script("training/train_svm.py", "TRAINING SVM"):
        sys.exit(1)
    
    # Step 3: Train LSTM
    if not run_script("training/train_lstm.py", "TRAINING LSTM"):
        sys.exit(1)
    
    # Step 4: Train PhoBERT
    if not run_script("training/train_phobert.py", "TRAINING PhoBERT"):
        sys.exit(1)
    
    # Step 5: Visualization & Summary Report
    print_step(5, 5, "VISUALIZATION & SUMMARY REPORT")
    print(f"\n{Colors.CYAN}Generating reports and visualization...{Colors.ENDC}\n")
    
    # Tạo Summary Report (file text)
    try:
        from utils.report_generator import generate_summary_report
        generate_summary_report()
        print(f"{Colors.GREEN}✓ Summary report created{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.WARNING}⚠️ Could not create summary report: {e}{Colors.ENDC}")
    
    # Visualization với matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend cho script
        import matplotlib.pyplot as plt
        
        from visualization.colab_visualize import (
            plot_class_distribution,
            plot_model_comparison,
            plot_training_history,
            plot_learning_curves_all
        )
        
        print(f"\n{Colors.CYAN}Generating charts...{Colors.ENDC}")
        
        # Lưu các biểu đồ vào file
        results_dir = ROOT_DIR / "results" / "charts"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Class Distribution
        try:
            plot_class_distribution()
            plt.savefig(results_dir / "class_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved class_distribution.png")
        except Exception as e:
            print(f"  ⚠️ class_distribution: {e}")
        
        # 2. Model Comparison
        try:
            plot_model_comparison()
            plt.savefig(results_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved model_comparison.png")
        except Exception as e:
            print(f"  ⚠️ model_comparison: {e}")
        
        # 3. Learning Curves
        try:
            plot_learning_curves_all()
            plt.savefig(results_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved learning_curves.png")
        except Exception as e:
            print(f"  ⚠️ learning_curves: {e}")
        
        print(f"\n{Colors.GREEN}✓ Charts saved to: {results_dir}{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.WARNING}⚠️ Could not run visualization: {e}{Colors.ENDC}")
    
    # Hiển thị tóm tắt kết quả
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}📊 RESULTS SUMMARY{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    
    try:
        import json
        saved_models_dir = ROOT_DIR / "saved_models"
        for model in ["svm", "lstm", "phobert"]:
            results_path = saved_models_dir / model / "results.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                test_r = results.get("test", results.get("validation", {}))
                acc = test_r.get("accuracy", 0)
                f1 = test_r.get("f1", 0)
                print(f"  {model.upper():10} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    except Exception as e:
        print(f"  Could not load results: {e}")
    
    print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.GREEN}✅ ALL STEPS COMPLETE!{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
    print(f"\n{Colors.CYAN}📁 Results saved to:{Colors.ENDC}")
    print(f"   - Reports: results/*.txt")
    print(f"   - Charts:  results/charts/*.png")
    print(f"   - Models:  saved_models/*/")
    print()


if __name__ == "__main__":
    main()
