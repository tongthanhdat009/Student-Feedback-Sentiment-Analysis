from fastapi import HTTPException


def test_account_controller_has_kaggle_auth_error_mapping():
    from pathlib import Path
    source = Path('app/controllers/account_controller.py').read_text(encoding='utf-8')
    assert "HTTPException(401" in source
    assert "Kaggle credentials unauthorized" in source
    assert "HTTPException(502" in source
