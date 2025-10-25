"""Database connection and session management"""
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
# Create this file:
# /c:/Users/aless/Documents/Projects/heimdall/services/data-ingestion-web/src/models/__init__.py

from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Connection string from environment
POSTGRES_USER = os.getenv("POSTGRES_USER", "heimdall_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "heimdall")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Engine configuration
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # No pooling (better for async)
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get a DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
