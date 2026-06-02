from pathlib import Path

EXPECTED = {'train': 11426, 'validation': 1583, 'test': 3166}
FILES = ['sents.txt', 'sentiments.txt', 'topics.txt']


def resolve_dataset_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[3]
    rooted = repo_root / path
    if rooted.exists():
        return rooted
    return candidate


def validate_local_uit_vsfc(path: str):
    root = resolve_dataset_path(path)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int] = {}
    if not root.exists() or not root.is_dir():
        return {'valid': False, 'errors': [f'Path not found: {path}'], 'warnings': warnings, 'stats': stats}
    for split, expected in EXPECTED.items():
        split_dir = root / split
        if not split_dir.exists():
            errors.append(f'Missing split directory: {split}')
            continue
        lines_by_file: dict[str, list[str]] = {}
        for name in FILES:
            file_path = split_dir / name
            if not file_path.exists():
                errors.append(f'Missing file: {split}/{name}')
                continue
            try:
                lines_by_file[name] = file_path.read_text(encoding='utf-8').splitlines()
            except UnicodeDecodeError:
                errors.append(f'File is not UTF-8: {split}/{name}')
            except Exception as exc:
                errors.append(f'Cannot read {split}/{name}: {exc}')
        if len(lines_by_file) != len(FILES):
            continue
        counts = {name: len(lines) for name, lines in lines_by_file.items()}
        if len(set(counts.values())) != 1:
            errors.append(f'Row count mismatch in {split}: {counts}')
            continue
        count = counts['sents.txt']
        stats[split] = count
        if count != expected:
            warnings.append(f'{split} row count {count} != expected {expected}')
        empty = [i + 1 for i, text in enumerate(lines_by_file['sents.txt']) if not text.strip()]
        if empty:
            errors.append(f'{split}/sents.txt has empty rows: {empty[:10]}')
        invalid_labels = sorted({label.strip() for label in lines_by_file['sentiments.txt'] if label.strip() not in {'0', '1', '2'}})
        if invalid_labels:
            errors.append(f'{split}/sentiments.txt has invalid labels: {invalid_labels[:10]}')
    return {'valid': not errors, 'errors': errors, 'warnings': warnings, 'stats': stats}
