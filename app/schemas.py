from pydantic import BaseModel, Field

class VoiceAnalysisRequest(BaseModel):
    language: str = "English"
    audioFormat: str = "mp3"
    audioBase64: str = Field(..., description="Base64 encoded audio string")

class VoiceClassification:
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class VoiceAnalysisResponse(BaseModel):
    status: str = "success"
    classification: str
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str = ""
