from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from backend.app.config import settings
from backend.app.database.connection import check_database_connection

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
    database_ok = check_database_connection()

    checks = {
        "api": "ok",
        "database": "ok" if database_ok else "failed",
        "redis": "not_checked_yet",
        "qdrant": "not_checked_yet"
    }

    is_ready = database_ok

    response_body = {
        "status": "ready" if is_ready else "not_ready",
        "service": settings.app_name,
        "environment": settings.environment,
        "checks": checks
    }

    return JSONResponse(
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response_body
    )