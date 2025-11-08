"""SQLAlchemy ORM models for Constellation entities (RBAC)."""

from datetime import datetime
from typing import Dict, Any, List
from uuid import UUID
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from .db import Base


class Constellation(Base):
    """
    Constellation - Logical grouping of WebSDR stations.
    
    A constellation represents a set of WebSDR receivers that can be used together
    for localization tasks. Constellations are owned by users and can be shared
    with other users with different permission levels (read/edit).
    """
    
    __tablename__ = "constellations"
    __table_args__ = {"schema": "heimdall"}
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Constellation details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Ownership (Keycloak user ID from JWT 'sub' claim)
    owner_id = Column(String(255), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    members = relationship("ConstellationMember", back_populates="constellation", cascade="all, delete-orphan")
    shares = relationship("ConstellationShare", back_populates="constellation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Constellation(id={self.id}, name='{self.name}', "
            f"owner_id='{self.owner_id}', members={len(self.members) if self.members else 0})>"
        )
    
    def to_dict(self, include_members: bool = False, include_shares: bool = False) -> Dict[str, Any]:
        """
        Convert constellation to dictionary for API responses.
        
        Args:
            include_members: Include list of WebSDR station IDs
            include_shares: Include list of shares (permissions)
        
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "member_count": len(self.members) if self.members else 0,
        }
        
        if include_members and self.members:
            result["members"] = [
                {
                    "websdr_station_id": str(member.websdr_station_id),
                    "added_at": member.added_at.isoformat() if member.added_at else None,
                    "added_by": member.added_by,
                }
                for member in self.members
            ]
        
        if include_shares and self.shares:
            result["shares"] = [
                {
                    "user_id": share.user_id,
                    "permission": share.permission,
                    "shared_by": share.shared_by,
                    "shared_at": share.shared_at.isoformat() if share.shared_at else None,
                }
                for share in self.shares
            ]
        
        return result


class ConstellationMember(Base):
    """
    ConstellationMember - Many-to-many relationship between Constellations and WebSDR Stations.
    
    Each row represents one WebSDR station belonging to one constellation.
    """
    
    __tablename__ = "constellation_members"
    __table_args__ = (
        UniqueConstraint('constellation_id', 'websdr_station_id', name='uq_constellation_websdr'),
        {"schema": "heimdall"}
    )
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Foreign keys
    constellation_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey('heimdall.constellations.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    websdr_station_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey('heimdall.websdr_stations.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    
    # Metadata
    added_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    added_by = Column(String(255), nullable=True)  # User who added this SDR
    
    # Relationships
    constellation = relationship("Constellation", back_populates="members")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ConstellationMember(id={self.id}, "
            f"constellation_id={self.constellation_id}, "
            f"websdr_station_id={self.websdr_station_id})>"
        )


class ConstellationShare(Base):
    """
    ConstellationShare - User access permissions for constellations.
    
    Defines which users can access a constellation and with what permission level:
    - 'read': View constellation and start localization sessions
    - 'edit': Modify constellation (add/remove SDRs, change name/description)
    
    The owner always has full control and doesn't need a share entry.
    """
    
    __tablename__ = "constellation_shares"
    __table_args__ = (
        UniqueConstraint('constellation_id', 'user_id', name='uq_constellation_user_share'),
        {"schema": "heimdall"}
    )
    
    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Foreign key to constellation
    constellation_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey('heimdall.constellations.id', ondelete='CASCADE'),
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
    
    # Relationships
    constellation = relationship("Constellation", back_populates="shares")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<ConstellationShare(id={self.id}, "
            f"constellation_id={self.constellation_id}, "
            f"user_id='{self.user_id}', permission='{self.permission}')>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert share to dictionary for API responses."""
        return {
            "id": str(self.id),
            "constellation_id": str(self.constellation_id),
            "user_id": self.user_id,
            "permission": self.permission,
            "shared_by": self.shared_by,
            "shared_at": self.shared_at.isoformat() if self.shared_at else None,
        }
