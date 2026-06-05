from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.core.db import check_db_connection

router = APIRouter()


@router.get("/health")
async def health():
    """
    Returns {"status": "ok"} if the database is reachable,
    {"status": "error"} otherwise.
    Never raises — all exceptions caught internally.
    """
    try:
        ok = check_db_connection()
        if ok:
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "error"})
    except Exception:
        return JSONResponse(status_code=503, content={"status": "error"})