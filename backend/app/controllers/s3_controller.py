from fastapi import APIRouter, Depends
from ..utils.auth import require_api_key
router=APIRouter(prefix='/api/kaggle/s3', tags=['s3'], dependencies=[Depends(require_api_key)])
