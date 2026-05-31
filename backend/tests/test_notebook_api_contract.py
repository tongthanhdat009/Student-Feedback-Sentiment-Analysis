from app.services.notebook_service import NotebookService


def test_service_validate_returns_invalid_for_missing_notebook(monkeypatch):
    svc = NotebookService(session=None)
    result = svc.validate('__missing__')
    assert result['valid'] is False
    assert result['errors']
