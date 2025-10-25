"""
Heimdall SDR - Common Authentication Module

This module provides centralized authentication and authorization
functionality for all Heimdall microservices using Keycloak.
"""

from .keycloak_auth import (
    KeycloakAuth,
    get_current_user,
    require_role,
    require_admin,
    require_operator
)
from .models import User, TokenData

__all__ = [
    "KeycloakAuth",
    "get_current_user",
    "require_role",
    "require_admin",
    "require_operator",
    "User",
    "TokenData",
]
