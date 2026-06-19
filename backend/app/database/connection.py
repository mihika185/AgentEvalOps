from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from backend.app.config import settings
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

class Base(DeclarativeBase):
    pass

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_database_connection() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    except SQLAlchemyError as exc:
        logger.warning("Database connection check failed: %s", exc)
        return False