import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from voicer.schemas import TTSRequest, VoiceCloneJSONRequest
from voicer.service import decode_base64_audio, get_service

app = FastAPI(title="voicer", version="1.0.0")


@app.on_event("startup")
def _load_model() -> None:
    get_service()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/tts", response_class=StreamingResponse)
async def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    service = get_service()
    wav_bytes, _sr = await run_in_threadpool(service.synthesize_to_wav_bytes, text)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=tts.wav"},
    )


@app.post("/voice-clone", response_class=StreamingResponse)
async def voice_clone(
    request: Request,
    text: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
    reference_text: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "")
    json_ref_bytes = None
    json_text = None
    json_reference_text = reference_text

    if content_type.startswith("application/json"):
        data = await request.json()
        payload = VoiceCloneJSONRequest(**data)
        json_text = payload.text
        json_ref_bytes = decode_base64_audio(payload.reference)
        json_reference_text = payload.reference_text

    effective_text = (json_text if json_text is not None else text) or ""
    effective_text = effective_text.strip()
    if not effective_text:
        raise HTTPException(status_code=400, detail="text is required")

    if json_ref_bytes is None:
        if reference is None:
            raise HTTPException(status_code=400, detail="reference file is required")
        ref_bytes = await reference.read()
        filename = reference.filename
    else:
        ref_bytes = json_ref_bytes
        filename = "reference.wav"

    if not ref_bytes:
        raise HTTPException(status_code=400, detail="reference file is empty")

    service = get_service()
    wav_bytes, _sr = await run_in_threadpool(
        service.voice_clone_to_wav_bytes,
        effective_text,
        ref_bytes,
        filename,
        json_reference_text,
    )
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=voice-clone.wav"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=7341)
