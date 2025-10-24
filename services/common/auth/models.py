"""
Authentication data models for Heimdall SDR services.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TokenData(BaseModel):
    """JWT token data extracted from validated token."""
    
    sub: str = Field(..., description="Subject (user ID)")
    preferred_username: Optional[str] = Field(None, description="Username")
    email: Optional[str] = Field(None, description="Email address")
    realm_roles: List[str] = Field(default_factory=list, description="Realm roles")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")


class User(BaseModel):
    """User information extracted from JWT token."""
    
    id: str = Field(..., description="User ID (sub claim)")
    username: Optional[str] = Field(None, description="Username")
    email: Optional[str] = Field(None, description="Email address")
    roles: List[str] = Field(default_factory=list, description="User roles")
    is_admin: bool = Field(default=False, description="Is user an admin")
    is_operator: bool = Field(default=False, description="Is user an operator")
    is_viewer: bool = Field(default=False, description="Is user a viewer")
    
    @classmethod
    def from_token_data(cls, token_data: TokenData) -> "User":
        """Create User from TokenData."""
        roles = token_data.realm_roles
        return cls(
            id=token_data.sub,
            username=token_data.preferred_username,
            email=token_data.email,
            roles=roles,
            is_admin="admin" in roles,
            is_operator="operator" in roles or "admin" in roles,
            is_viewer="viewer" in roles or "operator" in roles or "admin" in roles,
        )
