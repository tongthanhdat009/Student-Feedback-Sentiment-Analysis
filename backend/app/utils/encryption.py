from cryptography.fernet import Fernet, InvalidToken

class EncryptionService:
    def __init__(self, key: str):
        if key == 'generate-fernet-key':
            raise ValueError('FERNET_KEY must be set. Generate with Fernet.generate_key().decode().')
        self._fernet = Fernet(key.encode())
    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()
    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except InvalidToken as exc:
            raise ValueError('Invalid encrypted value') from exc
