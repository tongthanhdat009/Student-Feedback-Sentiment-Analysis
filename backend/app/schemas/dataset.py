from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class DatasetCreate(BaseModel):
    slug: str = Field(..., max_length=100)
    dataset_ref: str = Field(..., max_length=255)
    title: str | None = Field(None, max_length=255)
    description: str | None = None
    local_path: str | None = None
    status: str = 'active'


class DatasetUpdate(BaseModel):
    slug: str | None = Field(None, max_length=100)
    dataset_ref: str | None = Field(None, max_length=255)
    title: str | None = Field(None, max_length=255)
    description: str | None = None
    local_path: str | None = None
    status: str | None = None
    is_active: bool | None = None


class DatasetRead(BaseModel):
    id: UUID
    slug: str
    dataset_ref: str
    title: str | None = None
    description: str | None = None
    local_path: str | None = None
    status: str
    last_synced_at: datetime | None = None
    last_validated_at: datetime | None = None
    validation_result: dict | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    model_config = {'from_attributes': True}
