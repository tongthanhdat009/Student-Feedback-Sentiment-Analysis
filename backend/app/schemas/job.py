from datetime import datetime
from uuid import UUID
from pydantic import BaseModel
class NotebookTriggerRequest(BaseModel):
    account: str | None = None
    accounts: list[str] | None = None
    notebook_id: str
class JobRead(BaseModel):
    id: UUID
    account_id: UUID | None
    job_type: str
    target_ref: str
    status: str
    message: str | None = None
    output_path: str | None = None
    kaggle_ref: str | None = None
    staging_path: str | None = None
    last_polled_at: datetime | None = None
    timeout_seconds: int | None = None
    s3_object_key: str | None = None
    s3_presigned_url: str | None = None
    s3_presigned_url_expires_at: datetime | None = None
    staging_s3_prefix: str | None = None
    output_s3_prefix: str | None = None
    result_metadata: dict | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_config = {'from_attributes': True}
