from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class NotebookSyncRequest(BaseModel):
    account: str
    remote_slug: str | None = None
    title: str | None = None
    dataset_sources: list[str] | None = None
    enable_gpu: bool | None = None


class NotebookDeploymentRead(BaseModel):
    id: UUID
    account_id: UUID
    notebook_id: str
    kaggle_ref: str
    remote_slug: str
    remote_title: str | None = None
    source_path: str | None = None
    is_active: bool
    last_synced_at: datetime | None = None
    last_triggered_at: datetime | None = None
    last_status: str | None = None
    deployment_metadata: dict | None = None
    created_at: datetime
    updated_at: datetime
    model_config = {'from_attributes': True}
