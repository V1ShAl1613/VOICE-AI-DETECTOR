from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.core.config import settings
from app.ml.model import model_loader

app = FastAPI(
    title="Voice AI Detector API",
    description="API to detect AI-generated voices in multiple languages.",
    version="1.0.0"
)

# CORS configs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handlers for strict compliance
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Handle auth errors and other HTTP exceptions
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Handle schema validation errors (e.g. invalid base64, missing fields)
    return JSONResponse(
        status_code=400,
        content={
            "status": "error", 
            "message": f"Invalid request format: {str(exc.errors()[0].get('msg')) if exc.errors() else 'Unknown error'}"
        }
    )

@app.on_event("startup")
async def startup_event():
    # Ensure model is checked/loaded on startup
    # model_loader is a singleton instantiated on import, but we can re-check here
    print("----------------------------------------------------------------")
    print("DEPLOYMENT SIGNATURE: v2026-02-05-CORE-FEATURES-REFACTOR")
    print("If you see this, you are running the NEW code with core_features.py")
    print("----------------------------------------------------------------")
    if model_loader.model is None:
        print("WARNING: Model not loaded. API will return errors for predictions.")

app.include_router(routes.router, prefix="/api", tags=["Voice Detection"])

@app.get("/")
def root():
    return {"message": "Voice AI Detector API is running. Use POST /api/voice-detection to analyze audio."}
