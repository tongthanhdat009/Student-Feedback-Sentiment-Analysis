"""
Module quản lý cache cho models và datasets
============================================

Module này cung cấp các utility functions để:
1. Quản lý cache directory cho Huggingface models
2. Quản lý cache directory cho datasets
3. Load models từ cache hoặc download nếu chưa có
4. Kiểm tra trạng thái cache
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Thêm thư mục gốc vào path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
from datasets import load_dataset


def load_config() -> Dict[str, Any]:
    """Load cấu hình từ file YAML"""
    config_path = ROOT_DIR / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_cache_dir(config: Optional[Dict] = None) -> Path:
    """
    Lấy đường dẫn thư mục cache cho models
    
    Args:
        config: Cấu hình (nếu không cung cấp sẽ load từ file)
    
    Returns:
        Path đến thư mục cache
    """
    if config is None:
        config = load_config()
    
    cache_dir = ROOT_DIR / config["paths"]["model_cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable để Huggingface transformers sử dụng cache này
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    
    return cache_dir


def get_dataset_cache_dir(config: Optional[Dict] = None) -> Path:
    """
    Lấy đường dẫn thư mục cache cho datasets
    
    Args:
        config: Cấu hình (nếu không cung cấp sẽ load từ file)
    
    Returns:
        Path đến thư mục cache
    """
    if config is None:
        config = load_config()
    
    cache_dir = ROOT_DIR / config["paths"]["dataset_cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable để Huggingface datasets sử dụng cache này
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    
    return cache_dir


def get_saved_models_dir(config: Optional[Dict] = None) -> Path:
    """
    Lấy đường dẫn thư mục lưu models đã train
    
    Args:
        config: Cấu hình (nếu không cung cấp sẽ load từ file)
    
    Returns:
        Path đến thư mục saved_models
    """
    if config is None:
        config = load_config()
    
    saved_dir = ROOT_DIR / config["paths"]["saved_models"]
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    return saved_dir


def load_phobert_model(
    for_classification: bool = True,
    num_labels: int = 3,
    config: Optional[Dict] = None
):
    """
    Load PhoBERT model với caching
    Model sẽ được download một lần và lưu vào cache
    Các lần sau sẽ load từ cache
    
    Args:
        for_classification: Nếu True, load model cho classification
        num_labels: Số lượng labels (cho classification)
        config: Cấu hình
        
    Returns:
        tuple: (tokenizer, model)
    """
    if config is None:
        config = load_config()
    
    cache_dir = get_model_cache_dir(config)
    model_name = config["models"]["phobert"]["name"]
    
    print(f"📥 Loading PhoBERT model: {model_name}")
    print(f"   Cache directory: {cache_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        local_files_only=False  # Download nếu chưa có
    )
    print("   ✓ Tokenizer loaded")
    
    # Load model
    if for_classification:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            num_labels=num_labels,
            local_files_only=False
        )
        print(f"   ✓ Classification model loaded (num_labels={num_labels})")
    else:
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=False
        )
        print("   ✓ Base model loaded")
    
    return tokenizer, model


def load_cached_dataset(config: Optional[Dict] = None):
    """
    Load dataset từ cache
    Nếu chưa có trong cache, sẽ download
    
    Args:
        config: Cấu hình
        
    Returns:
        datasets.DatasetDict
    """
    if config is None:
        config = load_config()
    
    cache_dir = get_dataset_cache_dir(config)
    dataset_name = config["dataset"]["name"]
    
    print(f"📥 Loading dataset: {dataset_name}")
    print(f"   Cache directory: {cache_dir}")
    
    dataset = load_dataset(
        dataset_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True
    )
    
    print(f"   ✓ Dataset loaded with splits: {list(dataset.keys())}")
    
    return dataset


def check_cache_status(config: Optional[Dict] = None) -> Dict[str, bool]:
    """
    Kiểm tra trạng thái cache
    
    Args:
        config: Cấu hình
        
    Returns:
        Dictionary chứa trạng thái các cache
    """
    if config is None:
        config = load_config()
    
    model_cache = get_model_cache_dir(config)
    dataset_cache = get_dataset_cache_dir(config)
    saved_models = get_saved_models_dir(config)
    
    status = {
        "model_cache_exists": model_cache.exists() and any(model_cache.iterdir()),
        "dataset_cache_exists": dataset_cache.exists() and any(dataset_cache.iterdir()),
        "saved_models_exists": saved_models.exists() and any(saved_models.iterdir())
    }
    
    return status


def print_cache_info():
    """In thông tin về cache"""
    config = load_config()
    status = check_cache_status(config)
    
    print("\n" + "="*50)
    print("📁 CACHE STATUS")
    print("="*50)
    
    model_cache = get_model_cache_dir(config)
    dataset_cache = get_dataset_cache_dir(config)
    saved_models = get_saved_models_dir(config)
    
    print(f"\n1. Model Cache: {model_cache}")
    print(f"   Status: {'✓ Has content' if status['model_cache_exists'] else '✗ Empty'}")
    if model_cache.exists():
        files = list(model_cache.glob("**/*"))
        print(f"   Files: {len(files)}")
    
    print(f"\n2. Dataset Cache: {dataset_cache}")
    print(f"   Status: {'✓ Has content' if status['dataset_cache_exists'] else '✗ Empty'}")
    if dataset_cache.exists():
        files = list(dataset_cache.glob("**/*"))
        print(f"   Files: {len(files)}")
    
    print(f"\n3. Saved Models: {saved_models}")
    print(f"   Status: {'✓ Has content' if status['saved_models_exists'] else '✗ Empty'}")
    if saved_models.exists():
        for subdir in saved_models.iterdir():
            if subdir.is_dir():
                model_files = list(subdir.glob("*"))
                print(f"   - {subdir.name}/: {len(model_files)} files")
    
    print("\n" + "="*50)


def clear_cache(cache_type: str = "all", config: Optional[Dict] = None):
    """
    Xóa cache
    
    Args:
        cache_type: "model", "dataset", "saved", hoặc "all"
        config: Cấu hình
    """
    import shutil
    
    if config is None:
        config = load_config()
    
    if cache_type in ["model", "all"]:
        cache_dir = get_model_cache_dir(config)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared model cache: {cache_dir}")
    
    if cache_type in ["dataset", "all"]:
        cache_dir = get_dataset_cache_dir(config)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared dataset cache: {cache_dir}")
    
    if cache_type in ["saved", "all"]:
        saved_dir = get_saved_models_dir(config)
        if saved_dir.exists():
            shutil.rmtree(saved_dir)
            saved_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared saved models: {saved_dir}")


if __name__ == "__main__":
    # Test cache functions
    print_cache_info()
