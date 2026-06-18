from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.config import settings

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version
)

app.include_router(health_router, prefix=settings.api_prefix)

@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": "/docs"
    }