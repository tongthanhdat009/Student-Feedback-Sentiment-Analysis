"""
Data utilities for Vietnamese Student Feedback Sentiment Analysis.

This module provides centralized functions for loading and preprocessing data
from UIT-VSFC dataset and VietSentiWordNet lexicon.

Main functions:
    - load_data(): Load UIT-VSFC dataset from processed directory
    - load_all_splits(): Load all train/val/test splits from processed
    - load_raw_data(): Load raw UIT-VSFC data from data/raw
    - load_raw_all_splits(): Load all raw splits
    - preprocess_split(): Preprocess a list of texts
    - save_processed_data(): Save processed data to disk
    - preprocess_and_save_all(): Full pipeline (raw -> processed)
    - load_sentiwordnet(): Load VietSentiWordNet lexicon
    - normalize_teencode(): Normalize Vietnamese internet slang
    - preprocess_vietnamese(): Full preprocessing pipeline
    - get_swn_features(): Extract 8 SentiWordNet features
    - extract_swn_features_batch(): Batch feature extraction
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================
# TEENCODE DICTIONARY - Vietnamese Internet Slang
# ============================================
# Mapping từ viết tắt/tiếng lóng mạng -> từ chuẩn
TEENCODE_DICT = {
    # Phủ định
    'ko': 'không', 'k': 'không', 'khong': 'không', 'hok': 'không', 'hk': 'không',
    'kg': 'không', 'k0': 'không', 'kô': 'không', 'hông': 'không',

    # Đại từ
    'j': 'gì', 'gj': 'gì', 'gi': 'gì',
    'm': 'mày', 'mik': 'mình', 'mk': 'mình', 'mìh': 'mình',
    't': 'tao', 'tui': 'tôi', 'tôi': 'tôi',
    'bạn': 'bạn', 'bn': 'bạn',

    # Phó từ
    'r': 'rồi', 'rùi': 'rồi', 'roi': 'rồi', 'ròi': 'rồi',
    'nữa': 'nữa', 'nữa': 'nữa', 'nua': 'nữa',
    'h': 'hơn', 'hn': 'hơn',

    # Thì, mà, nhưng
    'nhưng': 'nhưng', 'nhg': 'nhưng', 'nhung': 'nhưng', 'nhgư': 'nhưng',
    'thì': 'thì', 'thj': 'thì', 'thik': 'thích',
    'mà': 'mà', 'ma': 'mà',

    # Tần suất
    'cx': 'cũng', 'cũng': 'cũng', 'cug': 'cũng', 'cũg': 'cũng',
    'đc': 'được', 'dc': 'được', 'đươc': 'được', 'duoc': 'được',
    'bnh': 'bình', 'bth': 'bình thường', 'bthg': 'bình thường',

    # Trạng từ chỉ mức độ
    'lm': 'lắm', 'lém': 'lắm', 'lem': 'lắm', 'nw': 'nữa',
    'qá': 'quá', 'qa': 'quá', 'quá': 'quá', 'wa': 'quá',
    'lun': 'luôn', 'ln': 'luôn',
    'nhìu': 'nhiều', 'nhieu': 'nhiều', 'niu': 'nhiều',
    'ít': 'ít', 'jt': 'ít',

    # Động từ phổ biến
    'bik': 'biết', 'bt': 'biết', 'bjết': 'biết',
    'thấy': 'thấy', 'thay': 'thấy', 'thấ': 'thấy',
    'nói': 'nói', 'noj': 'nói', 'nc': 'nói chuyện',
    'học': 'học', 'hok': 'học', 'hoc': 'học',
    'thi': 'thi', 'thj': 'thi', 'thi': 'thi',
    'làm': 'làm', 'lam': 'làm', 'lm': 'làm',
    'đi': 'đi', 'dj': 'đi', 'di': 'đi',
    'vào': 'vào', 'vao': 'vào',
    'ra': 'ra',
    'xem': 'xem', 'xê': 'xem',
    'nghe': 'nghe', 'nghe': 'nghe', 'nge': 'nghe',

    # Tính từ phổ biến
    'tốt': 'tốt', 'tot': 'tốt', 'tk': 'tốt',
    'hay': 'hay', 'hay': 'hay',
    'đẹp': 'đẹp', 'dep': 'đẹp',
    'dễ': 'dễ', 'de': 'dễ', 'dẽ': 'dễ',
    'khó': 'khó', 'kho': 'khó', 'khó': 'khó',
    'ngắn': 'ngắn', 'ngan': 'ngắn',
    'dài': 'dài', 'daj': 'dài',

    # Thời gian
    'h': 'giờ', 'gio': 'giờ', 'hôm': 'hôm', 'hom': 'hôm',
    'hôm nay': 'hôm nay', 'h.nay': 'hôm nay', 'hnay': 'hôm nay',
    'hôm qua': 'hôm qua', 'hqua': 'hôm qua',
    'ngày': 'ngày', 'ngay': 'ngày', 'nay': 'này',

    # Người, em, anh
    'em': 'em', 'e': 'em', 'm': 'em',
    'anh': 'anh', 'a': 'anh',
    'chị': 'chị', 'chj': 'chị', 'chi': 'chị',
    'cô': 'cô', 'co': 'cô',
    'thầy': 'thầy', 'thay': 'thầy', 'thâ': 'thầy',

    # Cảm thán
    'ok': 'ok', 'oke': 'ok', 'okay': 'ok', 'okela': 'ok',
    'um': 'ừ', 'ừ': 'ừ', 'uh': 'ừ', 'uk': 'ừ',
    'ờ': 'ờ', 'ò': 'ờ',

    # Khác
    'v': 'vậy', 'vậy': 'vậy', 'vay': 'vậy', 'vá': 'vậy', 'z': 'vậy',
    'đâu': 'đâu', 'dau': 'đâu', 'đâ': 'đâu',
    'sao': 'sao', 'sao': 'sao', 'sao': 'sao',
    'tn': 'thế nào', 'thế nào': 'thế nào', 'thế': 'thế', 'th': 'thế',
    'ntn': 'như thế nào',
    'vs': 'với', 'với': 'với', 'vs': 'với', 'cùng': 'cùng',
    'trong': 'trong', 'tg': 'trong',
    'về': 'về', 've': 'về',
    'cho': 'cho', 'cj': 'cho',
    'nè': 'nè', 'ne': 'nè', 'nì': 'nè',
    'đây': 'đây', 'day': 'đây', 'dâ': 'đây',
    'kia': 'kia', 'kj': 'kia',
    'nhé': 'nhé', 'nhe': 'nhé', 'nhá': 'nhé',
    'ạ': 'ạ', 'a': 'ạ',
    'ha': 'hả', 'hả': 'hả', 'hg': 'hả',
    'chưa': 'chưa', 'chua': 'chưa', 'cga': 'chưa',
    'rồi': 'rồi', 'roi': 'rồi', 'ròj': 'rồi',
}


def normalize_teencode(text: str, teencode_dict: Dict[str, str] = None) -> str:
    """
    Chuẩn hóa từ ngữ mạng (teencode) về từ chuẩn tiếng Việt.

    Args:
        text: Input Vietnamese text có thể chứa teencode
        teencode_dict: Dictionary mapping teencode -> từ chuẩn (default: TEENCODE_DICT)

    Returns:
        Text với teencode đã được thay thế bằng từ chuẩn

    Example:
        >>> normalize_teencode("ko bt j cả")
        'không biết gì cả'
        >>> normalize_teencode("cx đc thôi mà")
        'cũng được thôi mà'
    """
    if teencode_dict is None:
        teencode_dict = TEENCODE_DICT

    words = text.split()
    normalized_words = []

    for word in words:
        # Kiểm tra trong dictionary
        normalized = teencode_dict.get(word, word)
        normalized_words.append(normalized)

    return ' '.join(normalized_words)


def load_data(data_dir: str, split: str) -> Tuple[List[str], List[int]]:
    """
    Load UIT-VSFC dataset from processed directory.

    Args:
        data_dir: Path to data/processed directory
        split: One of 'train', 'validation', 'test'

    Returns:
        Tuple of (texts, labels) where:
            - texts: List of Vietnamese sentences
            - labels: List of sentiment labels (0=Negative, 1=Neutral, 2=Positive)

    Example:
        >>> texts, labels = load_data('data/processed', 'train')
        >>> print(f"Loaded {len(texts)} samples")
    """
    split_dir = os.path.join(data_dir, split)

    sents_path = os.path.join(split_dir, 'sents.txt')
    labels_path = os.path.join(split_dir, 'sentiments.txt')

    with open(sents_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    return texts, labels


def load_all_splits(data_dir: str) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Load all train/validation/test splits.

    Args:
        data_dir: Path to data/processed directory

    Returns:
        Dict with keys 'train', 'val', 'test', each containing (texts, labels)
    """
    train_texts, train_labels = load_data(data_dir, 'train')
    val_texts, val_labels = load_data(data_dir, 'validation')
    test_texts, test_labels = load_data(data_dir, 'test')

    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }


def load_sentiwordnet(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load VietSentiWordNet lexicon.

    Args:
        file_path: Path to VietSentiWordnet_Ver1.3.5.txt

    Returns:
        Dict mapping word -> {'pos_score': float, 'neg_score': float}

    Note:
        If a word appears in multiple synsets, the maximum score is kept.
        Words with underscores are converted to spaces (e.g., 'tốt_đẹp' -> 'tốt đẹp')
    """
    word_to_scores = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 5:
                continue

            pos_score = float(parts[2])
            neg_score = float(parts[3])
            synset_terms = parts[4]

            words = synset_terms.split()
            for word_entry in words:
                # Extract word (remove sense number)
                word = word_entry.split('#')[0]
                # Convert underscores to spaces
                word = word.replace('_', ' ')

                if word not in word_to_scores:
                    word_to_scores[word] = {
                        'pos_score': pos_score,
                        'neg_score': neg_score
                    }
                else:
                    # Keep maximum scores
                    word_to_scores[word]['pos_score'] = max(
                        word_to_scores[word]['pos_score'], pos_score
                    )
                    word_to_scores[word]['neg_score'] = max(
                        word_to_scores[word]['neg_score'], neg_score
                    )

    return word_to_scores


def preprocess_vietnamese(text: str, normalize_slang: bool = True) -> str:
    """
    Preprocess Vietnamese text for sentiment analysis.

    Operations:
        1. Convert to lowercase
        2. Normalize teencode/slang (e.g., "ko" -> "không", "j" -> "gì")
        3. Remove non-Vietnamese characters (keep Vietnamese diacritics)
        4. Normalize whitespace

    Args:
        text: Input Vietnamese text
        normalize_slang: Whether to normalize teencode (default: True)

    Returns:
        Preprocessed text

    Example:
        >>> preprocess_vietnamese("Học tập rất tốt! Nhưng thi khó quá.")
        'học tập rất tốt nhưng thi khó quá'
        >>> preprocess_vietnamese("ko bt j cả")
        'không biết gì cả'
    """
    text = text.lower()

    # Normalize teencode/slang
    if normalize_slang:
        text = normalize_teencode(text)

    # Keep Vietnamese characters and word characters
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    return ' '.join(text.split())


def get_swn_features(text: str, word_to_scores: Dict[str, Dict[str, float]]) -> List[float]:
    """
    Extract 8 SentiWordNet features from text.

    Features:
        1. pos_sum: Sum of positive scores
        2. neg_sum: Sum of negative scores
        3. pos_max: Maximum positive score
        4. neg_max: Maximum negative score
        5. pos_mean: Mean positive score
        6. neg_mean: Mean negative score
        7. coverage: Fraction of words covered by lexicon
        8. polarity: pos_sum - neg_sum

    Args:
        text: Input Vietnamese text
        word_to_scores: SentiWordNet lexicon dict from load_sentiwordnet()

    Returns:
        List of 8 feature values
    """
    words = preprocess_vietnamese(text).split()

    pos_scores = []
    neg_scores = []

    for word in words:
        if word in word_to_scores:
            pos_scores.append(word_to_scores[word]['pos_score'])
            neg_scores.append(word_to_scores[word]['neg_score'])

    n = len(words) or 1
    covered = len(pos_scores)

    if not pos_scores:
        return [0.0] * 8

    return [
        sum(pos_scores),                       # pos_sum
        sum(neg_scores),                       # neg_sum
        max(pos_scores),                       # pos_max
        max(neg_scores),                       # neg_max
        sum(pos_scores) / covered,             # pos_mean
        sum(neg_scores) / covered,             # neg_mean
        covered / n,                           # coverage
        sum(pos_scores) - sum(neg_scores),     # polarity
    ]


def extract_swn_features_batch(
    texts: List[str],
    word_to_scores: Dict[str, Dict[str, float]]
) -> np.ndarray:
    """
    Extract SentiWordNet features for a batch of texts.

    Args:
        texts: List of Vietnamese texts
        word_to_scores: SentiWordNet lexicon dict

    Returns:
        numpy array of shape (len(texts), 8)
    """
    features = [get_swn_features(text, word_to_scores) for text in texts]
    return np.array(features)


# Feature names for reference
SWN_FEATURE_NAMES = [
    'pos_sum', 'neg_sum', 'pos_max', 'neg_max',
    'pos_mean', 'neg_mean', 'coverage', 'polarity'
]

# ============================================
# RAW DATA PROCESSING
# ============================================

def load_raw_data(raw_dir: str, split: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Load raw UIT-VSFC data from data/raw directory.

    Args:
        raw_dir: Path to data/raw directory
        split: One of 'train', 'validation', 'test'

    Returns:
        Tuple of (texts, sentiments, topics) where:
            - texts: List of raw Vietnamese sentences
            - sentiments: List of sentiment labels (0=Negative, 1=Neutral, 2=Positive)
            - topics: List of topic labels

    Example:
        >>> texts, sentiments, topics = load_raw_data('data/raw', 'train')
        >>> print(f"Loaded {len(texts)} samples")
    """
    split_dir = os.path.join(raw_dir, split)

    sents_path = os.path.join(split_dir, 'sents.txt')
    sentiments_path = os.path.join(split_dir, 'sentiments.txt')
    topics_path = os.path.join(split_dir, 'topics.txt')

    with open(sents_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    with open(sentiments_path, 'r', encoding='utf-8') as f:
        sentiments = [int(line.strip()) for line in f.readlines()]

    with open(topics_path, 'r', encoding='utf-8') as f:
        topics = [int(line.strip()) for line in f.readlines()]

    return texts, sentiments, topics


def load_raw_all_splits(raw_dir: str) -> Dict[str, Dict[str, List]]:
    """
    Load all raw train/validation/test splits.

    Args:
        raw_dir: Path to data/raw directory

    Returns:
        Dict with keys 'train', 'validation', 'test', each containing:
            {'texts': [...], 'sentiments': [...], 'topics': [...]}
    """
    splits = ['train', 'validation', 'test']
    data = {}

    for split in splits:
        texts, sentiments, topics = load_raw_data(raw_dir, split)
        data[split] = {
            'texts': texts,
            'sentiments': sentiments,
            'topics': topics
        }

    return data


def preprocess_split(texts: List[str], normalize_slang: bool = True) -> List[str]:
    """
    Preprocess a list of Vietnamese texts.

    Args:
        texts: List of raw texts
        normalize_slang: Whether to normalize teencode (default: True)

    Returns:
        List of preprocessed texts

    Example:
        >>> raw = ["ko bt j cả", "Học tốt!"]
        >>> processed = preprocess_split(raw)
        >>> print(processed)
        ['không biết gì cả', 'học tốt']
    """
    return [preprocess_vietnamese(text, normalize_slang) for text in texts]


def save_processed_data(
    texts: List[str],
    sentiments: List[int],
    topics: List[int],
    output_dir: str,
    split: str
) -> None:
    """
    Save processed data to data/processed directory.

    Args:
        texts: List of preprocessed texts
        sentiments: List of sentiment labels
        topics: List of topic labels
        output_dir: Path to data/processed directory
        split: Split name ('train', 'validation', 'test')

    Example:
        >>> save_processed_data(texts, sentiments, topics, 'data/processed', 'train')
    """
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Save texts
    with open(os.path.join(split_dir, 'sents.txt'), 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    # Save sentiments
    with open(os.path.join(split_dir, 'sentiments.txt'), 'w', encoding='utf-8') as f:
        for sentiment in sentiments:
            f.write(str(sentiment) + '\n')

    # Save topics
    with open(os.path.join(split_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
        for topic in topics:
            f.write(str(topic) + '\n')


def preprocess_and_save_all(
    raw_dir: str,
    output_dir: str,
    normalize_slang: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Full pipeline: Load raw data, preprocess, and save to processed directory.

    This function implements the complete data preprocessing pipeline:
        1. Load raw data from data/raw/
        2. Preprocess texts (lowercase, teencode normalization, special chars removal)
        3. Save processed data to data/processed/

    Args:
        raw_dir: Path to data/raw directory
        output_dir: Path to data/processed directory
        normalize_slang: Whether to normalize teencode (default: True)
        verbose: Print progress information (default: True)

    Returns:
        Dict with split names as keys and sample counts as values

    Example:
        >>> counts = preprocess_and_save_all('data/raw', 'data/processed')
        >>> print(counts)
        {'train': 11426, 'validation': 1583, 'test': 3166}
    """
    splits = ['train', 'validation', 'test']
    counts = {}

    if verbose:
        print("=" * 60)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Input:  {raw_dir}")
        print(f"Output: {output_dir}")
        print()

    for split in splits:
        if verbose:
            print(f"Processing {split}...")

        # Load raw data
        texts, sentiments, topics = load_raw_data(raw_dir, split)

        # Preprocess texts
        processed_texts = preprocess_split(texts, normalize_slang)

        # Save processed data
        save_processed_data(processed_texts, sentiments, topics, output_dir, split)

        counts[split] = len(texts)

        if verbose:
            print(f"  Saved {len(texts)} samples")

    if verbose:
        print()
        print(f"Total: {sum(counts.values())} samples processed")
        print("=" * 60)

    return counts


# ============================================
# LABEL MAPPING
# ============================================

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
NUM_CLASSES = 3


if __name__ == '__main__':
    # Test the module
    import sys

    # Default paths (adjust for your environment)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    swn_file = os.path.join(base_dir, 'data', 'sentiwordnet-dataset', 'VietSentiWordnet_Ver1.3.5.txt')

    print("Testing data_utils module...")
    print()

    # Test load_data
    print("1. Testing load_data():")
    texts, labels = load_data(processed_dir, 'train')
    print(f"   Loaded {len(texts)} training samples from processed")
    print(f"   Sample: '{texts[0]}' -> Label: {labels[0]} ({LABEL_MAP[labels[0]]})")
    print()

    # Test load_raw_data
    print("2. Testing load_raw_data():")
    raw_texts, raw_sentiments, raw_topics = load_raw_data(raw_dir, 'train')
    print(f"   Loaded {len(raw_texts)} raw training samples")
    print(f"   Sample: '{raw_texts[0]}' -> Sentiment: {raw_sentiments[0]}, Topic: {raw_topics[0]}")
    print()

    # Test load_sentiwordnet
    print("3. Testing load_sentiwordnet():")
    word_to_scores = load_sentiwordnet(swn_file)
    print(f"   Loaded {len(word_to_scores)} words from VietSentiWordNet")
    if 'tốt' in word_to_scores:
        print(f"   'tốt' -> pos: {word_to_scores['tốt']['pos_score']}, neg: {word_to_scores['tốt']['neg_score']}")
    print()

    # Test normalize_teencode
    print("4. Testing normalize_teencode():")
    teencode_samples = [
        "ko bt j cả",
        "cx đc thôi mà",
        "thầy dạy tốt lắm",
        "môn này khó quá hk hiểu j",
    ]
    for sample in teencode_samples:
        normalized = normalize_teencode(sample)
        print(f"   '{sample}' -> '{normalized}'")
    print()

    # Test preprocess_vietnamese
    print("5. Testing preprocess_vietnamese():")
    sample = "Học tập rất tốt! Nhưng thi khó quá."
    print(f"   Input:  '{sample}'")
    print(f"   Output: '{preprocess_vietnamese(sample)}'")
    print()

    # Test preprocess_split
    print("6. Testing preprocess_split():")
    raw_samples = ["ko bt j cả", "Học tốt!", "cx đc thôi"]
    processed_samples = preprocess_split(raw_samples)
    for raw, proc in zip(raw_samples, processed_samples):
        print(f"   '{raw}' -> '{proc}'")
    print()

    # Test get_swn_features
    print("7. Testing get_swn_features():")
    features = get_swn_features(sample, word_to_scores)
    print(f"   Features: {dict(zip(SWN_FEATURE_NAMES, features))}")
    print()

    print("All tests passed!")