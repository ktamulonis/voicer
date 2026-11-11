from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")


class VoiceCloneJSONRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    reference: str = Field(..., min_length=1, description="Base64-encoded WAV audio")
    reference_text: str | None = Field(None, description="Exact transcript of the reference audio for higher-fidelity cloning")
