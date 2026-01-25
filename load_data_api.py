# ============================================
# LOAD UIT-VSFC DATASET USING API
# ============================================
# Hướng dẫn 3 cách load dataset qua API

import os
import requests
import pandas as pd
import json
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÁCH 1: SỬ DỤNG HUGGINGFACE API (Recommended)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset_via_huggingface_api():
    """
    Load dataset từ Huggingface Hub sử dụng API
    URL: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
    """
    from datasets import load_dataset

    print("📥 Loading UIT-VSFC dataset from Huggingface API...")
    print("   API: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback\n")

    # Load dataset
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")

    print("✅ Dataset loaded successfully!")
    print(f"\n📊 Dataset structure:")
    print(dataset)

    # Convert to DataFrames
    train_df = pd.DataFrame(dataset['train'])
    validation_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    print(f"\n📈 Dataset statistics:")
    print(f"   • Training samples: {len(train_df):,}")
    print(f"   • Validation samples: {len(validation_df):,}")
    print(f"   • Test samples: {len(test_df):,}")
    print(f"   • Total samples: {len(train_df) + len(validation_df) + len(test_df):,}")

    return train_df, validation_df, test_df, dataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÁCH 2: TẢI DATASET TỪ GITHUB (Manual Download)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset_from_github():
    """
    Tải dataset từ GitHub repository của UIT
    Repository: https://github.com/uit-nlp/vietnamese-students-feedback
    """
    import urllib.request

    print("📥 Downloading UIT-VSFC dataset from GitHub...")
    print("   Repository: https://github.com/uit-nlp/vietnamese-students-feedback\n")

    # URLs cho dataset
    base_url = "https://raw.githubusercontent.com/uit-nlp/vietnamese-students-feedback/master/data"

    files = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv'
    }

    data_dir = Path('data/uit-vsfc')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download files
    dfs = {}
    for split, filename in files.items():
        url = f"{base_url}/{filename}"
        filepath = data_dir / filename

        print(f"📥 Downloading {filename}...")

        try:
            urllib.request.urlretrieve(url, filepath)
            df = pd.read_csv(filepath)
            dfs[split] = df
            print(f"   ✅ {filename}: {len(df):,} samples")
        except Exception as e:
            print(f"   ❌ Error downloading {filename}: {e}")

    print(f"\n✅ Dataset saved to: {data_dir}")

    return dfs.get('train'), dfs.get('val'), dfs.get('test')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÁCH 3: SỬ DỤNG HUGGINGFACE REST API (Direct HTTP Request)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset_via_huggingface_rest_api():
    """
    Sử dụng Huggingface REST API để tải dataset
    Không cần cài đặt thư viện datasets
    """
    print("📥 Loading UIT-VSFC dataset via Huggingface REST API...\n")

    # Huggingface API endpoint
    dataset_name = "uitnlp/vietnamese_students_feedback"
    api_base = "https://huggingface.co/api/datasets"

    # Try to get dataset info
    try:
        response = requests.get(f"{api_base}/{dataset_name}")
        response.raise_for_status()

        data = response.json()
        print("✅ Dataset info retrieved!")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # Download individual files
        repo_id = dataset_name.replace('/', '--')
        files_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"

        splits = ['train', 'validation', 'test']
        dfs = {}

        data_dir = Path('data/uit-vsfc-api')
        data_dir.mkdir(parents=True, exist_ok=True)

        for split in splits:
            filename = f"{split}.parquet"  # hoặc .json
            url = f"{files_url}/{filename}"
            filepath = data_dir / filename

            print(f"\n📥 Downloading {filename}...")

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Read file
                if filename.endswith('.parquet'):
                    import pyarrow.parquet as pq
                    table = pq.read_table(filepath)
                    df = table.to_pandas()
                elif filename.endswith('.json'):
                    df = pd.read_json(filepath, lines=True)
                else:
                    df = pd.read_csv(filepath)

                dfs[split] = df
                print(f"   ✅ {filename}: {len(df):,} samples")

            except Exception as e:
                print(f"   ⚠️ Could not download {filename}: {e}")

        return dfs.get('train'), dfs.get('validation'), dfs.get('test')

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Recommend: Use CÁCH 1 (Huggingface datasets library)")
        return None, None, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CÁCH 4: TỰ TẠO DATASET TỪ LOCAL FILES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_dataset_from_local(data_path='data/uit-vsfc'):
    """
    Load dataset từ local files đã tải trước đó
    """
    data_path = Path(data_path)

    print(f"📂 Loading dataset from local: {data_path}\n")

    if not data_path.exists():
        print(f"❌ Path không tồn tại: {data_path}")
        return None, None, None

    dfs = {}
    for file in data_path.glob('*.csv'):
        split_name = file.stem  # train, val, test
        df = pd.read_csv(file)
        dfs[split_name] = df
        print(f"   ✅ {file.name}: {len(df):,} samples")

    return dfs.get('train'), dfs.get('val'), dfs.get('test')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN: CHỌN CÁCH LOAD DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Chọn cách load dataset:
    LOAD_METHOD = "huggingface"  # Options: "huggingface", "github", "api", "local"

    if LOAD_METHOD == "huggingface":
        # ✅ CÁCH 1: Dễ nhất, khuyên dùng
        train_df, val_df, test_df, dataset = load_dataset_via_huggingface_api()

    elif LOAD_METHOD == "github":
        # ⚠️ Cần kiểm tra URL có còn hoạt động không
        train_df, val_df, test_df = load_dataset_from_github()

    elif LOAD_METHOD == "api":
        # ⚠️ Cách phức tạp hơn, không cần thư viện datasets
        train_df, val_df, test_df = load_dataset_via_huggingface_rest_api()

    elif LOAD_METHOD == "local":
        # 📂 Load từ files đã tải sẵn
        train_df, val_df, test_df = load_dataset_from_local()

    # Hiển thị kết quả
    if train_df is not None:
        print("\n" + "="*60)
        print("📊 FINAL DATASET SUMMARY")
        print("="*60)
        print(f"   • Training: {len(train_df):,} samples")
        print(f"   • Validation: {len(val_df):,} samples")
        print(f"   • Test: {len(test_df):,} samples")
        print(f"   • TOTAL: {len(train_df) + len(val_df) + len(test_df):,} samples")
        print("="*60)

        # Display sample
        print("\n📋 Sample data:")
        print(train_df.head())
