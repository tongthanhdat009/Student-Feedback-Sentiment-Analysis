import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..database import Base

def utcnow(): return datetime.now(timezone.utc)

class KaggleJob(Base):
    __tablename__ = 'kaggle_jobs'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('kaggle_accounts.id'), nullable=True)
    job_type: Mapped[str] = mapped_column(String(50))
    target_ref: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default='pending')
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    kaggle_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    staging_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    timeout_seconds: Mapped[int | None] = mapped_column(nullable=True)
    s3_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    s3_presigned_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    s3_presigned_url_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    staging_s3_prefix: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_s3_prefix: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    account = relationship('KaggleAccount', back_populates='jobs')
