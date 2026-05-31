import os

from app.services.kaggle_client_factory import KaggleClientFactory


class FakeApi:
    seen = {}
    def authenticate(self):
        FakeApi.seen = {
            'KAGGLE_USERNAME': os.environ.get('KAGGLE_USERNAME'),
            'KAGGLE_KEY': os.environ.get('KAGGLE_KEY'),
            'KAGGLE_API_TOKEN': os.environ.get('KAGGLE_API_TOKEN'),
        }


def test_factory_sets_legacy_key_and_new_access_token(monkeypatch):
    monkeypatch.setattr('app.services.kaggle_client_factory.KaggleApi', FakeApi)
    monkeypatch.delenv('KAGGLE_USERNAME', raising=False)
    monkeypatch.delenv('KAGGLE_KEY', raising=False)
    monkeypatch.delenv('KAGGLE_API_TOKEN', raising=False)

    KaggleClientFactory().create('user1', 'secret-token-or-key')

    assert FakeApi.seen == {
        'KAGGLE_USERNAME': 'user1',
        'KAGGLE_KEY': 'secret-token-or-key',
        'KAGGLE_API_TOKEN': 'secret-token-or-key',
    }
    assert os.environ.get('KAGGLE_USERNAME') is None
    assert os.environ.get('KAGGLE_KEY') is None
    assert os.environ.get('KAGGLE_API_TOKEN') is None


def test_factory_restores_previous_env(monkeypatch):
    monkeypatch.setattr('app.services.kaggle_client_factory.KaggleApi', FakeApi)
    monkeypatch.setenv('KAGGLE_USERNAME', 'old-user')
    monkeypatch.setenv('KAGGLE_KEY', 'old-key')
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'old-token')

    KaggleClientFactory().create('new-user', 'new-key')

    assert os.environ['KAGGLE_USERNAME'] == 'old-user'
    assert os.environ['KAGGLE_KEY'] == 'old-key'
    assert os.environ['KAGGLE_API_TOKEN'] == 'old-token'
