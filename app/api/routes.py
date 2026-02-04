from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.schemas import VoiceAnalysisRequest, VoiceAnalysisResponse, VoiceClassification, ErrorResponse
from app.core.security import get_api_key
from app.audio.decoder import decode_audio, cleanup_temp_dir
from app.audio.features import extract_features
from app.ml.model import model_loader
from app.ml.explanation import generate_explanation
from app.core.config import settings

router = APIRouter()

@router.post("/voice-detection", response_model=VoiceAnalysisResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_voice(request: VoiceAnalysisRequest, api_key: str = Depends(get_api_key)):
    """
    Analyzes the provided audio to detect if it is AI-generated or Human.
    """
    temp_file_path = None
    temp_dir = None
    try:
        # 1. Decode Audio
        temp_file_path, temp_dir = decode_audio(request.audioBase64)
        
        # 2. Extract Features
        try:
            features = extract_features(temp_file_path)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Audio processing failed: {str(e)}"}
            )
            
        # 3. Predict
        try:
            # Score is probability of being AI (class 1)
            score = model_loader.predict(features)
        except RuntimeError as e:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Model inference failed: {str(e)}"}
            )
            
        # 4. Classify & Explain
        # Clamp score between 0.02 and 0.98 (updated range for better granularity)
        score = max(0.02, min(0.98, score))
        
        threshold = settings.AI_PROBABILITY_THRESHOLD
        
        if score >= threshold:
            classification = VoiceClassification.AI_GENERATED
        else:
            classification = VoiceClassification.HUMAN
            
        explanation = generate_explanation(features, score, threshold)
        
        return VoiceAnalysisResponse(
            language=request.language,
            classification=classification,
            confidenceScore=round(score, 4),
            explanation=explanation
        )
        
    finally:
        # Cleanup - safe for Windows
        if temp_dir:
            cleanup_temp_dir(temp_dir)

