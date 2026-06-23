from fastapi import FastAPI

from backend.app.api.chunks import router as chunks_router
from backend.app.api.documents import router as documents_router
from backend.app.api.health import router as health_router
from backend.app.api.rag import router as rag_router
from backend.app.api.retrieval import router as retrieval_router
from backend.app.config import settings
from backend.app.logging_config import configure_logging, get_logger
from backend.app.api.runs import router as runs_router
from backend.app.api.quality_gates import router as quality_gates_router

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(chunks_router, prefix=settings.api_prefix)
app.include_router(retrieval_router, prefix=settings.api_prefix)
app.include_router(rag_router, prefix=settings.api_prefix)
app.include_router(runs_router, prefix=settings.api_prefix)
app.include_router(quality_gates_router, prefix=settings.api_prefix)

logger.info(
    "%s initialized in %s mode",
    settings.app_name,
    settings.environment
)

@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": "/docs"
    }