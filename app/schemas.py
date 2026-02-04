from pydantic import BaseModel, Field, field_validator
from enum import Enum

class Language(str, Enum):
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"

class AudioFormat(str, Enum):
    MP3 = "mp3"

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

class VoiceAnalysisRequest(BaseModel):
    language: Language
    audioFormat: AudioFormat = Field(default=AudioFormat.MP3)
    audioBase64: str = Field(..., description="Base64 encoded audio string")
    
    @field_validator('language', mode='before')
    @classmethod
    def strip_language(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class VoiceClassification(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class VoiceAnalysisResponse(BaseModel):
    status: str = "success"
    language: Language
    classification: VoiceClassification
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
