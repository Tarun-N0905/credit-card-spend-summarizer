import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from src.api.v1.ingestion.ingestion import run_ingestion

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=422,
            content={"message": "Only PDF files are accepted."},
        )

    tmp_path: Path | None = None

    try:
        suffix = Path(file.filename).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        result = run_ingestion(
            file_path=str(tmp_path),
            original_filename=file.filename,
        )

        return result

    except Exception:
        return JSONResponse(
            status_code=500,
            content=_UNAVAILABLE,
        )

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
