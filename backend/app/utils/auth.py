from fastapi import Header, HTTPException, status
from ..config import get_settings
async def require_api_key(x_api_key: str | None = Header(default=None, alias='X-API-Key')):
    if x_api_key != get_settings().admin_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid API key')
