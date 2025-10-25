"""Type compatibility for common patterns."""

from typing import TypeVar, Protocol, Union, Any
from pathlib import Path

T = TypeVar("T")
U = TypeVar("U")


class Closeable(Protocol):
    """Protocol for closeable resources."""

    def close(self) -> None:
        """Close the resource."""
        ...


class Serializable(Protocol):
    """Protocol for JSON-serializable objects."""

    def dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...

    def json(self) -> str:
        """Convert to JSON string."""
        ...


JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]
PathLike = Union[str, Path]
