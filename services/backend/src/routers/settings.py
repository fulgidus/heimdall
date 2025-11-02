"""
User settings router for Heimdall SDR backend.

Provides endpoints for managing user settings.
"""

import logging
import os

from fastapi import APIRouter, Depends, status

from ..db import get_pool
from ..models.settings import (
    UserSettings,
    UserSettingsCreate,
    UserSettingsUpdate,
)

# Import authentication from common module
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from common.auth.keycloak_auth import get_current_user
from common.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


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
