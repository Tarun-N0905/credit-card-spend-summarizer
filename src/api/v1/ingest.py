import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from src.ingestion.ingestion import run_ingestion

router = APIRouter()

_UNAVAILABLE = {"message": "Service temporarily unavailable. Please try again later."}


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload.

    Flow:
      1. Stream the uploaded file to a secure temp file on disk
      2. Call run_ingestion(file_path) from ingestion.py
      3. Return {status, doc_id, chunks_ingested, chunks_skipped}
      4. Clean up the temp file regardless of success or failure

    Only PDF files are accepted. All exceptions return the generic safe error.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=422,
            content={"message": "Only PDF files are accepted."},
        )

    tmp_path: Path | None = None
    try:
        # Write upload to a named temp file so ingestion.py can open it by path
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        result = run_ingestion(str(tmp_path))
        return result

    except Exception:
        return JSONResponse(status_code=500, content=_UNAVAILABLE)

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)