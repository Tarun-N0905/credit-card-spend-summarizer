from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.v1.routes.router import router as v1_router

app = FastAPI(
    title="Credit Card Spend Summarizer",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Service temporarily unavailable. Please try again later."},
    )


app.include_router(v1_router)
