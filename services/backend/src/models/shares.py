"""SQLAlchemy ORM models for sharing resources (RBAC)."""

from datetime import datetime
from typing import Dict, Any
from uuid import UUID
from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from .db import Base


class SourceShare(Base):
    """
    SourceShare - User access permissions for known radio sources.
    
    Defines which users can access a source and with what permission level:
    - 'read': View source details (frequency, location, etc.)
    - 'edit': Modify source (update frequency, location, power, etc.)
    
    The owner always has full control and doesn't need a share entry.
    Public sources (is_public=true) are visible to all users without a share.
    """
    
    __tablename__ = "source_shares"
    __table_args__ = (
        UniqueConstraint('source_id', 'user_id', name='uq_source_user_share'),
        {"schema": "heimdall"}
    )
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Foreign key to known_sources
    source_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey('heimdall.known_sources.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # User being granted access (Keycloak user ID)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Permission level: 'read' or 'edit'
    permission = Column(String(20), nullable=False, index=True)
    
    # Metadata
    shared_by = Column(String(255), nullable=False)  # User who created this share
    shared_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<SourceShare(id={self.id}, "
            f"source_id={self.source_id}, "
            f"user_id='{self.user_id}', permission='{self.permission}')>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert share to dictionary for API responses."""
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "user_id": self.user_id,
            "permission": self.permission,
            "shared_by": self.shared_by,
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
        }


class ModelShare(Base):
    """
    ModelShare - User access permissions for ML models.
    
    Defines which users can access a model and with what permission level:
    - 'read': View model metadata and use for inference
    - 'edit': Modify model metadata, retrain, or delete
    
    The owner always has full control and doesn't need a share entry.
    """
    
    __tablename__ = "model_shares"
    __table_args__ = (
        UniqueConstraint('model_id', 'user_id', name='uq_model_user_share'),
        {"schema": "heimdall"}
    )
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Foreign key to models
    model_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey('heimdall.models.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # User being granted access (Keycloak user ID)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Permission level: 'read' or 'edit'
    permission = Column(String(20), nullable=False, index=True)
    
    # Metadata
    shared_by = Column(String(255), nullable=False)  # User who created this share
    shared_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ModelShare(id={self.id}, "
            f"model_id={self.model_id}, "
            f"user_id='{self.user_id}', permission='{self.permission}')>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert share to dictionary for API responses."""
        return {
            "id": str(self.id),
            "model_id": str(self.model_id),
            "user_id": self.user_id,
            "permission": self.permission,
            "shared_by": self.shared_by,
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
        }
