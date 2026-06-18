import logging
import sys
from backend.app.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_log_level() -> int:
    if settings.debug:
        return logging.DEBUG

    return logging.INFO

def configure_logging() -> None:
    logging.basicConfig(
        level=get_log_level(),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)