from datetime import datetime, timezone
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_session
from ..repositories.job_repository import JobRepository
from ..schemas.job import JobRead
from ..utils.auth import require_api_key
from ..services.job_worker import worker
from ..services.s3_service import S3Service
router=APIRouter(prefix='/api/kaggle/jobs', tags=['jobs'], dependencies=[Depends(require_api_key)])
@router.get('')
async def list_jobs(page: int = 1, page_size: int = 20, session: AsyncSession=Depends(get_session)):
    repo = JobRepository(session); start = (page-1)*page_size
    items = await repo.list(limit=page_size, offset=start)
    total = await repo.count()
    return {'items': items, 'total': total, 'page': page, 'page_size': page_size}
@router.get('/{job_id}', response_model=JobRead)
async def get_job(job_id: UUID, session: AsyncSession=Depends(get_session)):
    job=await JobRepository(session).get(job_id)
    if not job: raise HTTPException(404, 'Job not found')
    return job
@router.post('/{job_id}/download-output', response_model=JobRead)
async def download(job_id: UUID, session: AsyncSession=Depends(get_session)):
    repo=JobRepository(session); job=await repo.get(job_id)
    if not job: raise HTTPException(404, 'Job not found')
    if job.status != 'completed': raise HTTPException(409, {'error': 'Kaggle run not ready', 'status': job.status})
    if not job.kaggle_ref: raise HTTPException(409, {'error': 'Kaggle ref missing'})
    job.status='pending'; job.job_type='notebook_output_download'; await repo.save(job); worker.enqueue_download(job.id); return job
@router.get('/{job_id}/artifact-url')
async def artifact_url(job_id: UUID, session: AsyncSession=Depends(get_session)):
    repo=JobRepository(session); job=await repo.get(job_id)
    if not job or not job.s3_object_key: raise HTTPException(404, 'Artifact not found')
    if not job.s3_presigned_url or not job.s3_presigned_url_expires_at or job.s3_presigned_url_expires_at <= datetime.now(timezone.utc):
        job.s3_presigned_url, job.s3_presigned_url_expires_at = S3Service().presign_get(job.s3_object_key); await repo.save(job)
    return {'url': job.s3_presigned_url, 'expires_at': job.s3_presigned_url_expires_at}
