from pydantic import BaseModel, Field
from enum import Enum

class Language(str, Enum):
    TAMIL = "Tamil"
    ENGLISH = "English"
    HINDI = "Hindi"
    MALAYALAM = "Malayalam"
    TELUGU = "Telugu"

class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"

class VoiceAnalysisRequest(BaseModel):
    language: Language
    audioFormat: AudioFormat = Field(default=AudioFormat.MP3)
    audioBase64: str = Field(..., description="Base64 encoded audio string")

class VoiceClassification(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"

class VoiceAnalysisResponse(BaseModel):
    status: str = "success"
    language: Language
    classification: VoiceClassification
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
