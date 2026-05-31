from app.services.kaggle_quota_service import QUOTA_ENDPOINTS, fetch_kaggle_quota


def test_quota_service_has_candidate_endpoints():
    assert QUOTA_ENDPOINTS
    assert all(url.startswith('https://www.kaggle.com/') for url in QUOTA_ENDPOINTS)


def test_quota_service_returns_unavailable_when_requests_fail(monkeypatch):
    class Boom:
        status_code = 500
        def raise_for_status(self):
            raise RuntimeError('nope')
    monkeypatch.setattr('app.services.kaggle_quota_service.requests.get', lambda *a, **k: Boom())
    result = fetch_kaggle_quota('u', 'k')
    assert result['available'] is False
    assert result['message']
