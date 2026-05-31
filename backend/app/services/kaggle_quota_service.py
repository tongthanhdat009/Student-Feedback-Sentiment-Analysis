import requests


QUOTA_ENDPOINTS = [
    'https://www.kaggle.com/api/i/users/quota',
    'https://www.kaggle.com/api/i/users/current/quota',
    'https://www.kaggle.com/api/i/compute/userquota',
]


def fetch_kaggle_quota(username: str, token_or_key: str) -> dict:
    """Best-effort Kaggle quota lookup.

    Kaggle's public Python SDK exposes no quota method in current installed version.
    Try known web/API shapes, return unavailable instead of failing account page.
    """
    errors: list[str] = []
    auth = (username, token_or_key)
    headers = {'Authorization': f'Bearer {token_or_key}', 'Accept': 'application/json'}
    for url in QUOTA_ENDPOINTS:
        try:
            res = requests.get(url, headers=headers, auth=auth, timeout=10)
            if res.status_code == 404:
                errors.append(f'{url}: 404')
                continue
            res.raise_for_status()
            data = res.json()
            return {'available': True, 'source': url, 'raw': data}
        except Exception as exc:
            errors.append(f'{url}: {exc}')
    return {
        'available': False,
        'source': None,
        'raw': None,
        'message': 'Kaggle quota endpoint is not exposed by the installed Kaggle SDK or available API endpoints.',
        'errors': errors[-3:],
    }
