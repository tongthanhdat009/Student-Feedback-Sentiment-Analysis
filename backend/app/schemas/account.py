from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field
class AccountCreate(BaseModel):
    name: str = Field(..., max_length=100)
    kaggle_username: str
    kaggle_key: str
class AccountRead(BaseModel):
    id: UUID
    name: str
    kaggle_username: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None
    model_config = {'from_attributes': True}
