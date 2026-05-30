from pathlib import Path

def safe_child(base: str | Path, *parts: str) -> Path:
    root = Path(base).resolve()
    target = root.joinpath(*parts).resolve()
    if root != target and root not in target.parents:
        raise ValueError('Path traversal blocked')
    return target

def normalize_s3_key(key: str) -> str:
    if key.startswith('/') or '\\' in key or '..' in key.split('/'):
        raise ValueError('Invalid S3 object key')
    if not key.startswith('kaggle-outputs/'):
        raise ValueError('S3 object key must stay under kaggle-outputs/')
    return key
