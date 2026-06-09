from fastapi import APIRouter

from src.api.v1.routes.health import router as health_router
from src.api.v1.routes.chat import router as chat_router
from src.api.v1.routes.ingest import router as ingest_router
from src.api.v1.routes.conversations import router as conversations_router

router = APIRouter()

# Health check lives at root level
router.include_router(health_router)

# All feature routes under /api/v1
router.include_router(chat_router, prefix="/api/v1")
router.include_router(ingest_router, prefix="/api/v1")
router.include_router(conversations_router, prefix="/api/v1")
