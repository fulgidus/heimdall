"""
Keycloak authentication handler for Heimdall SDR services.

This module provides JWT validation and user authentication using Keycloak
as the identity provider.
"""

import os
import logging
from typing import Optional, List
from functools import wraps

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import TokenData, User

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class KeycloakAuth:
    """
    Keycloak authentication handler.
    
    Validates JWT tokens issued by Keycloak and extracts user information.
    """
    
    def __init__(
        self,
        keycloak_url: Optional[str] = None,
        realm: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """
        Initialize Keycloak authentication handler.
        
        Args:
            keycloak_url: Keycloak server URL (default: from env KEYCLOAK_URL)
            realm: Keycloak realm name (default: from env KEYCLOAK_REALM)
            client_id: Client ID for this service (default: from env)
        """
        self.keycloak_url = keycloak_url or os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
        self.realm = realm or os.getenv("KEYCLOAK_REALM", "heimdall")
        self.client_id = client_id or os.getenv("KEYCLOAK_CLIENT_ID", "api-gateway")
        
        # Construct JWKS URL for token validation
        self.jwks_url = f"{self.keycloak_url}/realms/{self.realm}/protocol/openid-connect/certs"
        
        # Initialize JWK client for fetching public keys
        self.jwk_client = PyJWKClient(self.jwks_url)
        
        logger.info(f"Initialized KeycloakAuth with realm={self.realm}, jwks_url={self.jwks_url}")
    
    def verify_token(self, token: str) -> TokenData:
        """
        Verify JWT token and extract claims.
        
        Args:
            token: JWT token string
            
        Returns:
            TokenData with extracted claims
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Get signing key from JWKS
            signing_key = self.jwk_client.get_signing_key_from_jwt(token)
            
            # Decode and verify token
            # Note: Public clients don't have aud claim by default
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": False,  # Public clients may not have aud claim
                }
            )
            
            # Extract realm roles from token
            realm_access = payload.get("realm_access", {})
            realm_roles = realm_access.get("roles", [])
            
            # Create TokenData
            token_data = TokenData(
                sub=payload.get("sub"),
                preferred_username=payload.get("preferred_username"),
                email=payload.get("email"),
                realm_roles=realm_roles,
                exp=payload.get("exp"),
                iat=payload.get("iat"),
            )
            
            logger.debug(f"Token verified for user: {token_data.preferred_username}")
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def get_current_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> User:
        """
        FastAPI dependency to get current authenticated user.
        
        Args:
            credentials: HTTP Bearer credentials from request
            
        Returns:
            User object with user information
            
        Raises:
            HTTPException: If authentication fails
        """
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = credentials.credentials
        token_data = self.verify_token(token)
        user = User.from_token_data(token_data)
        
        return user


# Global auth instance (initialized on first use)
_auth_instance: Optional[KeycloakAuth] = None


def get_auth() -> KeycloakAuth:
    """Get or create global KeycloakAuth instance."""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = KeycloakAuth()
    return _auth_instance


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    FastAPI dependency to get current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"message": f"Hello {user.username}"}
    """
    auth = get_auth()
    return await auth.get_current_user(credentials)


def require_role(required_roles: List[str]):
    """
    FastAPI dependency factory to require specific roles.
    
    Usage:
        @app.get("/admin-only")
        async def admin_route(user: User = Depends(require_role(["admin"]))):
            return {"message": "Admin access granted"}
    
    Args:
        required_roles: List of required roles (user must have at least one)
        
    Returns:
        FastAPI dependency function
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        # Check if user has any of the required roles
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {required_roles}",
            )
        return user
    
    return role_checker


def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to require admin role.
    
    Usage:
        @app.delete("/models/{model_id}")
        async def delete_model(model_id: str, user: User = Depends(require_admin)):
            # Only admins can delete models
            ...
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def require_operator(user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to require operator role (or admin).
    
    Usage:
        @app.post("/signals/acquisition")
        async def trigger_acquisition(user: User = Depends(require_operator)):
            # Operators and admins can trigger acquisition
            ...
    """
    if not user.is_operator:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator access required",
        )
    return user
