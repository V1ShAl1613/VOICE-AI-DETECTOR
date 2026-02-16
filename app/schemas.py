from typing import Optional
from pydantic import BaseModel, Field, model_validator

class VoiceAnalysisRequest(BaseModel):
    language: str = "English"
    audioFormat: str = "mp3"
    audioBase64: Optional[str] = Field(None, description="Base64 encoded audio string")
    audioUrl: Optional[str] = Field(None, description="URL to download audio file from")

    @model_validator(mode='after')
    def check_audio_source(self):
        if not self.audioBase64 and not self.audioUrl:
            raise ValueError('Either audioBase64 or audioUrl must be provided')
        return self

class VoiceClassification:
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class VoiceAnalysisResponse(BaseModel):
    status: str = "success"
    classification: str
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str = ""
