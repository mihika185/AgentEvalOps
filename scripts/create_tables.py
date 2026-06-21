from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import inspect
from backend.app.database.connection import Base, engine
from backend.app.logging_config import configure_logging, get_logger
from backend.app.database import models

configure_logging()
logger = get_logger(__name__)

def create_tables() -> None:
    logger.info("Creating database tables...")

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    logger.info("Database tables created successfully.")
    logger.info("Available tables: %s", ", ".join(table_names))

if __name__ == "__main__":
    create_tables()