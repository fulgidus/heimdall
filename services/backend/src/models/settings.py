"""
User settings models for Heimdall SDR backend.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class UserSettingsBase(BaseModel):
    """Base model for user settings."""

    # General settings
    theme: str = Field(default="dark", pattern="^(dark|light|auto)$")
    language: str = Field(default="en", pattern="^(en|it)$")
    timezone: str = Field(default="UTC")
    auto_refresh: bool = Field(default=True)
    refresh_interval: int = Field(default=30, ge=10, le=300)

    # API settings
    api_timeout: int = Field(default=30000, ge=5000, le=60000)
    retry_attempts: int = Field(default=3, ge=0, le=5)
    enable_caching: bool = Field(default=True)

    # Notification settings
    email_notifications: bool = Field(default=True)
    system_alerts: bool = Field(default=True)
    performance_warnings: bool = Field(default=True)
    webhook_url: Optional[str] = Field(default=None, max_length=512)

    # Advanced settings
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="info", pattern="^(error|warn|info|debug)$")
    max_concurrent_requests: int = Field(default=5, ge=1, le=10)

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate webhook URL format."""
        if v and v.strip():
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("Webhook URL must start with http:// or https://")
            return v.strip()
        return None


class UserSettingsCreate(UserSettingsBase):
    """Model for creating user settings."""

    user_id: str = Field(..., description="Keycloak user ID")


class UserSettingsUpdate(BaseModel):
    """Model for updating user settings (all fields optional)."""

    # General settings
    theme: Optional[str] = Field(default=None, pattern="^(dark|light|auto)$")
    language: Optional[str] = Field(default=None, pattern="^(en|it)$")
    timezone: Optional[str] = None
    auto_refresh: Optional[bool] = None
    refresh_interval: Optional[int] = Field(default=None, ge=10, le=300)

    # API settings
    api_timeout: Optional[int] = Field(default=None, ge=5000, le=60000)
    retry_attempts: Optional[int] = Field(default=None, ge=0, le=5)
    enable_caching: Optional[bool] = None

    # Notification settings
    email_notifications: Optional[bool] = None
    system_alerts: Optional[bool] = None
    performance_warnings: Optional[bool] = None
    webhook_url: Optional[str] = Field(default=None, max_length=512)

    # Advanced settings
    debug_mode: Optional[bool] = None
    log_level: Optional[str] = Field(default=None, pattern="^(error|warn|info|debug)$")
    max_concurrent_requests: Optional[int] = Field(default=None, ge=1, le=10)

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate webhook URL format."""
        if v and v.strip():
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("Webhook URL must start with http:// or https://")
            return v.strip()
        return None


class UserSettings(UserSettingsBase):
    """Complete user settings model with database fields."""

    id: UUID
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PasswordChangeRequest(BaseModel):
    """Request model for password change."""

    current_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8)


class UsernameChangeRequest(BaseModel):
    """Request model for username change."""

    new_username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
