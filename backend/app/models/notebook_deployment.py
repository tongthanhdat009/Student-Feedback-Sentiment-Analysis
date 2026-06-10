import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..database import Base


def utcnow(): return datetime.now(timezone.utc)


class NotebookDeployment(Base):
    __tablename__ = 'notebook_deployments'
    __table_args__ = (
        UniqueConstraint('account_id', 'notebook_id', name='uq_notebook_deployments_account_notebook'),
        UniqueConstraint('account_id', 'kaggle_ref', name='uq_notebook_deployments_account_ref'),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('kaggle_accounts.id'), nullable=False)
    notebook_id: Mapped[str] = mapped_column(String(255), nullable=False)
    kaggle_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_slug: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    deployment_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    account = relationship('KaggleAccount')
