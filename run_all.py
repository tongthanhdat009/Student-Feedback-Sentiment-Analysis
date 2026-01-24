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
    
    # Step 5: Run Dashboard
    print_step(5, 5, "LAUNCHING DASHBOARD")
    print(f"\n{Colors.CYAN}Dashboard starting at: http://localhost:8501{Colors.ENDC}")
    print(f"{Colors.WARNING}Press Ctrl+C to stop{Colors.ENDC}\n")
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        cwd=str(ROOT_DIR)
    )
    
    print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.GREEN}COMPLETE!{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")


if __name__ == "__main__":
    main()
