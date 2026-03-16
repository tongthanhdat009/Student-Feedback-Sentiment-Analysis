"""
Source package for Vietnamese Student Feedback Sentiment Analysis.
"""

from .data_utils import (
    load_data,
    load_all_splits,
    load_sentiwordnet,
    preprocess_vietnamese,
    get_swn_features,
    extract_swn_features_batch,
    SWN_FEATURE_NAMES,
    LABEL_MAP,
    NUM_CLASSES,
)

__all__ = [
    'load_data',
    'load_all_splits',
    'load_sentiwordnet',
    'preprocess_vietnamese',
    'get_swn_features',
    'extract_swn_features_batch',
    'SWN_FEATURE_NAMES',
    'LABEL_MAP',
    'NUM_CLASSES',
]