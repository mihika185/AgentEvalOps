from fastapi import APIRouter
from backend.app.config import settings

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)

@router.get("")
def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }

@router.get("/readiness")
def readiness_check():
    return {
        "status": "ready",
        "checks": {
            "api": "ok",
            "database": "not_configured_yet",
            "redis": "not_configured_yet",
            "qdrant": "not_configured_yet"
        }
    }