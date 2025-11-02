"""
User profile API endpoints.

Handles user profile CRUD operations complementing Keycloak authentication.
"""

import logging
import sys
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

# Add parent directory to path for common module
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from common.auth.keycloak_auth import get_current_user
from common.auth.models import User

from ..db import get_db_session
from ..models.user import UserProfile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/users", tags=["users"])


# Pydantic models for request/response
class UserProfileUpdate(BaseModel):
    """User profile update request."""

    first_name: str | None = Field(None, max_length=255)
    last_name: str | None = Field(None, max_length=255)
    phone: str | None = Field(None, max_length=50)
    organization: str | None = Field(None, max_length=255)
    location: str | None = Field(None, max_length=255)
    bio: str | None = None


class UserProfileResponse(BaseModel):
    """User profile response with Keycloak data merged."""

    user_id: str
    username: str | None
    email: str | None
    roles: list[str]
    first_name: str | None
    last_name: str | None
    phone: str | None
    organization: str | None
    location: str | None
    bio: str | None
    created_at: str | None
    updated_at: str | None


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """
    Get current user's profile.
    
    Merges Keycloak user data (from JWT) with extended profile data from database.
    """
    async with get_db_session() as session:
        # Query for existing profile
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()

        # Build response merging Keycloak data with profile data
        return UserProfileResponse(
            user_id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            roles=current_user.roles,
            first_name=profile.first_name if profile else None,
            last_name=profile.last_name if profile else None,
            phone=profile.phone if profile else None,
            organization=profile.organization if profile else None,
            location=profile.location if profile else None,
            bio=profile.bio if profile else None,
            created_at=profile.created_at.isoformat() if profile and profile.created_at else None,
            updated_at=profile.updated_at.isoformat() if profile and profile.updated_at else None,
        )


@router.put("/me", response_model=UserProfileResponse)
async def update_current_user_profile(
    profile_update: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """
    Update current user's profile.
    
    Creates profile if it doesn't exist, updates if it does.
    Only updates fields that are provided in the request.
    """
    async with get_db_session() as session:
        # Query for existing profile
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == current_user.id)
        )
        profile = result.scalar_one_or_none()

        if profile is None:
            # Create new profile
            profile = UserProfile(
                user_id=current_user.id,
                first_name=profile_update.first_name,
                last_name=profile_update.last_name,
                phone=profile_update.phone,
                organization=profile_update.organization,
                location=profile_update.location,
                bio=profile_update.bio,
            )
            session.add(profile)
            logger.info(f"Created new profile for user {current_user.id}")
        else:
            # Update existing profile (only fields that are provided)
            if profile_update.first_name is not None:
                profile.first_name = profile_update.first_name
            if profile_update.last_name is not None:
                profile.last_name = profile_update.last_name
            if profile_update.phone is not None:
                profile.phone = profile_update.phone
            if profile_update.organization is not None:
                profile.organization = profile_update.organization
            if profile_update.location is not None:
                profile.location = profile_update.location
            if profile_update.bio is not None:
                profile.bio = profile_update.bio

            profile.updated_at = datetime.utcnow()
            logger.info(f"Updated profile for user {current_user.id}")

        await session.commit()
        await session.refresh(profile)

        # Return merged response
        return UserProfileResponse(
            user_id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            roles=current_user.roles,
            first_name=profile.first_name,
            last_name=profile.last_name,
            phone=profile.phone,
            organization=profile.organization,
            location=profile.location,
            bio=profile.bio,
            created_at=profile.created_at.isoformat() if profile.created_at else None,
            updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
        )


@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: str,
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """
    Get another user's profile.
    
    Note: This endpoint returns public profile information only.
    Some fields may be restricted based on user roles.
    """
    async with get_db_session() as session:
        # Query for profile
        result = await session.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        profile = result.scalar_one_or_none()

        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"User profile not found: {user_id}"
            )

        # Build response (without Keycloak data since we don't have their JWT)
        return UserProfileResponse(
            user_id=profile.user_id,
            username=None,  # Not available without Keycloak lookup
            email=None,  # Privacy: don't expose other users' emails
            roles=[],  # Not available without Keycloak lookup
            first_name=profile.first_name,
            last_name=profile.last_name,
            phone=None,  # Privacy: don't expose phone numbers
            organization=profile.organization,
            location=profile.location,
            bio=profile.bio,
            created_at=profile.created_at.isoformat() if profile.created_at else None,
            updated_at=profile.updated_at.isoformat() if profile.updated_at else None,
        )
