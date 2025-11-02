"""
User settings router for Heimdall SDR backend.

Provides endpoints for managing user settings and Keycloak user profile updates.
"""

import logging
import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from ..db import get_pool
from ..models.settings import (
    PasswordChangeRequest,
    UserSettings,
    UserSettingsCreate,
    UserSettingsUpdate,
    UsernameChangeRequest,
)

# Import authentication from common module
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from common.auth.keycloak_auth import get_current_user
from common.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


# Keycloak Admin API configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "heimdall")
KEYCLOAK_ADMIN_CLIENT_ID = os.getenv("KEYCLOAK_API_GATEWAY_CLIENT_ID", "api-gateway")
KEYCLOAK_ADMIN_CLIENT_SECRET = os.getenv(
    "KEYCLOAK_API_GATEWAY_CLIENT_SECRET", "api-gateway-secret"
)


async def get_admin_token() -> str:
    """Get Keycloak admin access token using client credentials."""
    token_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": KEYCLOAK_ADMIN_CLIENT_ID,
                "client_secret": KEYCLOAK_ADMIN_CLIENT_SECRET,
            },
        )

        if response.status_code != 200:
            logger.error(f"Failed to get admin token: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to authenticate with identity provider",
            )

        return response.json()["access_token"]


@router.get("/", response_model=UserSettings)
async def get_user_settings(user: User = Depends(get_current_user)) -> UserSettings:
    """
    Get current user's settings.

    Creates default settings if none exist.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Try to fetch existing settings
        row = await conn.fetchrow(
            """
            SELECT * FROM heimdall.user_settings
            WHERE user_id = $1
            """,
            user.id,
        )

        if row:
            return UserSettings(**dict(row))

        # Create default settings if none exist
        row = await conn.fetchrow(
            """
            INSERT INTO heimdall.user_settings (user_id)
            VALUES ($1)
            RETURNING *
            """,
            user.id,
        )

        return UserSettings(**dict(row))


@router.put("/", response_model=UserSettings)
async def update_user_settings(
    settings_update: UserSettingsUpdate, user: User = Depends(get_current_user)
) -> UserSettings:
    """
    Update current user's settings.

    Only provided fields will be updated.
    """
    pool = await get_pool()

    # Build dynamic update query based on provided fields
    update_fields = []
    values = []
    param_counter = 1

    for field, value in settings_update.model_dump(exclude_unset=True).items():
        update_fields.append(f"{field} = ${param_counter}")
        values.append(value)
        param_counter += 1

    if not update_fields:
        # No fields to update, just return current settings
        return await get_user_settings(user)

    # Add user_id as final parameter
    values.append(user.id)

    query = f"""
        UPDATE heimdall.user_settings
        SET {', '.join(update_fields)}
        WHERE user_id = ${param_counter}
        RETURNING *
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *values)

        if not row:
            # Settings don't exist, create them first
            await conn.execute(
                """
                INSERT INTO heimdall.user_settings (user_id)
                VALUES ($1)
                """,
                user.id,
            )
            # Try update again
            row = await conn.fetchrow(query, *values)

        return UserSettings(**dict(row))


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    password_request: PasswordChangeRequest, user: User = Depends(get_current_user)
) -> None:
    """
    Change user's password via Keycloak Admin API.

    Requires current password for verification.
    """
    # First, verify current password by attempting to get a token
    token_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"

    async with httpx.AsyncClient() as client:
        # Verify current password
        response = await client.post(
            token_url,
            data={
                "grant_type": "password",
                "client_id": KEYCLOAK_ADMIN_CLIENT_ID,
                "username": user.username or user.email,
                "password": password_request.current_password,
            },
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect",
            )

        # Get admin token to update password
        admin_token = await get_admin_token()

        # Update password via Keycloak Admin API
        user_url = f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user.id}"
        reset_password_url = f"{user_url}/reset-password"

        response = await client.put(
            reset_password_url,
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "type": "password",
                "value": password_request.new_password,
                "temporary": False,
            },
        )

        if response.status_code != 204:
            logger.error(f"Failed to update password: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password",
            )

    logger.info(f"Password changed successfully for user {user.id}")


@router.post("/username", status_code=status.HTTP_204_NO_CONTENT)
async def change_username(
    username_request: UsernameChangeRequest, user: User = Depends(get_current_user)
) -> None:
    """
    Change user's username via Keycloak Admin API.

    Requires admin privileges.
    """
    # Get admin token
    admin_token = await get_admin_token()

    async with httpx.AsyncClient() as client:
        # Update username via Keycloak Admin API
        user_url = f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users/{user.id}"

        response = await client.put(
            user_url,
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"username": username_request.new_username},
        )

        if response.status_code == 409:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Username already exists"
            )
        elif response.status_code not in [200, 204]:
            logger.error(f"Failed to update username: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update username",
            )

    logger.info(f"Username changed successfully for user {user.id} to {username_request.new_username}")


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def reset_user_settings(user: User = Depends(get_current_user)) -> None:
    """
    Reset user settings to defaults.

    Deletes current settings; they will be recreated with defaults on next GET.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM heimdall.user_settings
            WHERE user_id = $1
            """,
            user.id,
        )

    logger.info(f"Settings reset to defaults for user {user.id}")
