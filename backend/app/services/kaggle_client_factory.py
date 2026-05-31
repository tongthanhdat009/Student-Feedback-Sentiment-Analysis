import os, threading
from kaggle.api.kaggle_api_extended import KaggleApi

_lock = threading.Lock()


class KaggleClientFactory:
    def create(self, username: str, key: str) -> KaggleApi:
        with _lock:
            old_user = os.environ.get('KAGGLE_USERNAME')
            old_key = os.environ.get('KAGGLE_KEY')
            old_token = os.environ.get('KAGGLE_API_TOKEN')
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
            os.environ['KAGGLE_API_TOKEN'] = key
            try:
                api = KaggleApi(); api.authenticate(); return api
            finally:
                if old_user is None: os.environ.pop('KAGGLE_USERNAME', None)
                else: os.environ['KAGGLE_USERNAME'] = old_user
                if old_key is None: os.environ.pop('KAGGLE_KEY', None)
                else: os.environ['KAGGLE_KEY'] = old_key
                if old_token is None: os.environ.pop('KAGGLE_API_TOKEN', None)
                else: os.environ['KAGGLE_API_TOKEN'] = old_token
