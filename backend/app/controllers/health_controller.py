from fastapi import APIRouter
router=APIRouter(prefix='/api/kaggle', tags=['health'])
@router.get('/health')
async def health(): return {'ok': True, 'service': 'kaggle-notebook-manager'}
