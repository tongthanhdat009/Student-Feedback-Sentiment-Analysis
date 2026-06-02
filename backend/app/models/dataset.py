import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


def utcnow(): return datetime.now(timezone.utc)


class KaggleDataset(Base):
    __tablename__ = 'kaggle_datasets'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    dataset_ref: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default='active')
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    validation_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
