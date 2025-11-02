"""User profile data models for Heimdall SDR services."""

from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from .db import Base


class UserProfile(Base):
    """
    User profile information extending Keycloak user data.
    
    Stores additional profile fields not managed by Keycloak:
    - Phone number
    - Organization affiliation
    - Location
    - Bio/description
    
    Primary authentication and user data (email, username, roles) comes from Keycloak JWT.
    """

    __tablename__ = "user_profiles"
    __table_args__ = {"schema": "heimdall"}

    # Primary key - matches Keycloak user ID (sub claim from JWT)
    user_id = Column(String(255), primary_key=True)

    # Extended profile fields
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    organization = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<UserProfile(user_id='{self.user_id}', name='{self.first_name} {self.last_name}')>"

    def to_dict(self) -> dict[str, Any]:
        """Convert user profile to dictionary for API responses."""
        return {
            "user_id": self.user_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phone": self.phone,
            "organization": self.organization,
            "location": self.location,
            "bio": self.bio,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
