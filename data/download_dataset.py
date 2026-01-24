"""
Download UIT-VSFC Dataset từ Huggingface Parquet API
=====================================================

Sử dụng Parquet files từ Huggingface để download dữ liệu thực
"""

import os
import sys
from pathlib import Path
import json
import requests
import pandas as pd
import io

# Thêm thư mục gốc vào path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import yaml

# Load config
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"

# Huggingface Parquet API
DATASET_NAME = "uitnlp/vietnamese_students_feedback"
PARQUET_BASE = "https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default"


def load_config():
    """Load cấu hình từ file yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config):
    """Tạo các thư mục cần thiết"""
    dirs = [
        config["paths"]["model_cache"],
        config["paths"]["dataset_cache"],
        config["paths"]["saved_models"],
        config["paths"]["raw_data"],
        config["paths"]["processed_data"],
        config["paths"]["results"],
    ]
    
    for d in dirs:
        path = ROOT_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Đã tạo/kiểm tra thư mục: {path}")


def get_parquet_urls():
    """Lấy danh sách Parquet file URLs"""
    print(f"\n📡 Fetching Parquet file list...")
    
    splits = ["train", "validation", "test"]
    parquet_urls = {}
    
    for split in splits:
        url = f"{PARQUET_BASE}/{split}"
        print(f"   Checking {split}...")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                files = response.json()
                if files:
                    parquet_urls[split] = files
                    print(f"   ✓ {split}: {len(files)} file(s)")
        except Exception as e:
            print(f"   ⚠️ Error getting {split}: {e}")
    
    return parquet_urls


def download_parquet_split(split_name, parquet_url):
    """Download và parse Parquet file"""
    print(f"\n📥 Downloading {split_name} from Parquet...")
    print(f"   URL: {parquet_url}")
    
    try:
        response = requests.get(parquet_url, timeout=120)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        # Read Parquet từ bytes
        import pyarrow.parquet as pq
        
        buffer = io.BytesIO(response.content)
        table = pq.read_table(buffer)
        df = table.to_pandas()
        
        print(f"   ✓ Downloaded {len(df)} samples")
        
        # Convert to records
        records = df.to_dict('records')
        return records, df
        
    except ImportError:
        print("   ⚠️ pyarrow not installed. Trying pandas...")
        
        # Fallback: lưu file và đọc bằng pandas
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            f.write(response.content)
            temp_path = f.name
        
        df = pd.read_parquet(temp_path)
        os.unlink(temp_path)
        
        print(f"   ✓ Downloaded {len(df)} samples")
        records = df.to_dict('records')
        return records, df
        
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return [], None


def download_all_data(config):
    """Download toàn bộ dataset từ Parquet"""
    raw_data_dir = ROOT_DIR / config["paths"]["raw_data"]
    cache_dir = ROOT_DIR / config["paths"]["dataset_cache"]
    
    print(f"\n{'='*60}")
    print(f"📥 DOWNLOADING UIT-VSFC DATASET FROM PARQUET FILES")
    print(f"{'='*60}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Save location: {raw_data_dir}")
    
    # Get Parquet URLs
    parquet_urls = get_parquet_urls()
    
    if not parquet_urls:
        print("\n⚠️ Could not get Parquet URLs. Trying direct download...")
        # Direct Parquet URLs (fallback)
        parquet_urls = {
            "train": ["https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/main/data/train-00000-of-00001.parquet"],
            "validation": ["https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/main/data/validation-00000-of-00001.parquet"],
            "test": ["https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/resolve/main/data/test-00000-of-00001.parquet"]
        }
    
    all_data = {}
    
    for split_name, urls in parquet_urls.items():
        if isinstance(urls, list) and urls:
            url = urls[0] if isinstance(urls[0], str) else urls[0].get('url', urls[0])
        else:
            url = urls
            
        try:
            records, df = download_parquet_split(split_name, url)
            
            if records:
                # Save to CSV
                csv_path = raw_data_dir / f"{split_name}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"   ✓ Saved to: {csv_path}")
                
                all_data[split_name] = records
            
        except Exception as e:
            print(f"   ⚠️ Error downloading {split_name}: {e}")
    
    if all_data:
        # Save combined data to cache
        cache_path = cache_dir / "uit_vsfc_data.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved cache to: {cache_path}")
        
        # Create HuggingFace Dataset
        try:
            from datasets import Dataset, DatasetDict
            
            dataset_dict = {}
            for split_name, records in all_data.items():
                if records:
                    dataset_dict[split_name] = Dataset.from_list(records)
            
            if dataset_dict:
                dataset = DatasetDict(dataset_dict)
                dataset_save_path = cache_dir / "uit_vsfc_dataset"
                dataset.save_to_disk(str(dataset_save_path))
                print(f"✓ Saved HuggingFace Dataset to: {dataset_save_path}")
        except Exception as e:
            print(f"⚠️ Could not create HuggingFace Dataset: {e}")
    
    return all_data


def show_sample_data(data):
    """Hiển thị mẫu dữ liệu"""
    print(f"\n{'='*60}")
    print("📝 SAMPLE DATA")
    print(f"{'='*60}")
    
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    topic_labels = {0: "Lecturer", 1: "Training Program", 2: "Facility", 3: "Others"}
    
    train_data = data.get("train", [])
    
    if train_data:
        print("\n🔤 First 5 samples from train set:")
        for i, sample in enumerate(train_data[:5]):
            print(f"\n--- Sample {i+1} ---")
            sentence = sample.get('sentence', 'N/A')
            print(f"  Sentence: {sentence[:80]}...")
            sentiment = sample.get('sentiment', 0)
            topic = sample.get('topic', 0)
            print(f"  Sentiment: {sentiment} ({sentiment_labels.get(sentiment, 'Unknown')})")
            print(f"  Topic: {topic} ({topic_labels.get(topic, 'Unknown')})")
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    for split_name, records in data.items():
        if records:
            df = pd.DataFrame(records)
            print(f"\n  {split_name.upper()}: {len(records)} samples")
            
            if 'sentiment' in df.columns:
                print("    Sentiment distribution:")
                for val, count in df['sentiment'].value_counts().sort_index().items():
                    label = sentiment_labels.get(val, f"Unknown({val})")
                    pct = count / len(df) * 100
                    print(f"      {val} ({label}): {count} ({pct:.1f}%)")


def main():
    """Main function"""
    print(f"\n{'#'*60}")
    print("# UIT-VSFC DATASET DOWNLOADER (Parquet API)")
    print(f"{'#'*60}")
    
    # Load config
    print("\n📋 Loading configuration...")
    config = load_config()
    print(f"✓ Config loaded from: {CONFIG_PATH}")
    
    # Create directories
    print("\n📁 Creating directories...")
    ensure_directories(config)
    
    # Download data
    data = download_all_data(config)
    
    if data:
        # Show sample
        show_sample_data(data)
        
        print(f"\n{'='*60}")
        print("🎉 DATASET DOWNLOAD COMPLETE!")
        print(f"{'='*60}")
        
        total_samples = sum(len(records) for records in data.values())
        print(f"\n📊 Total samples downloaded: {total_samples}")
        
        print("\n📁 Data saved to:")
        print(f"   CSV files: {ROOT_DIR / config['paths']['raw_data']}")
        print(f"   Cache: {ROOT_DIR / config['paths']['dataset_cache']}")
    else:
        print("\n❌ Failed to download dataset!")
    
    print("\n🚀 Next steps:")
    print("   python preprocessing/build_dataset.py")
    print("   python training/train_svm.py")


if __name__ == "__main__":
    main()
