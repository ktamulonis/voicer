# voicer

Local text-to-speech and voice cloning microservice using Qwen3-TTS.

## Status

- ✅ Qwen3-TTS runs locally on macOS (CPU)
- ✅ Produces WAV output via Python API
- ✅ FastAPI microservice on `http://localhost:7341`
- ⚠️ Uses compatibility shims for Hugging Face internals
- ⚠️ Experimental / research-grade

## Install

```bash
pip install -r qwen3_tts_space/requirements.txt
pip install fastapi uvicorn python-multipart
```

## Quick test (Python API)

```bash
python - <<EOF
from qwen3_tts import tts
tts("Hello from voicer")
EOF
```

## Start the server

```bash
python server.py
```

Server listens on `http://localhost:7341`.

## Health check

```bash
curl http://localhost:7341/health
```

## TTS via curl

```bash
curl -X POST http://localhost:7341/tts \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello, this is local Qwen3 text to speech"}' \\
  --output out.wav
```

## Voice clone via curl

```bash
curl -X POST http://localhost:7341/voice-clone \\
  -F "text=Hello, this is my cloned voice" \\
  -F "reference=@ref.wav" \\
  --output clone.wav
```

## Voice Cloning Quality Notes

- Use a clean, close-mic recording with minimal background noise.
- Keep reference audio between ~5–20 seconds for best speaker identity.
- Preprocessing applies gentle band-limiting and RMS normalization to reduce metallic artifacts.
- If output still sounds buzzy, try a different reference with less sibilance.

## Rails-friendly usage

- JSON TTS: `POST /tts` with `{"text":"..."}`; response is `audio/wav`.
- Multipart voice clone: `POST /voice-clone` with `text` + `reference` file; response is `audio/wav`.
- No base64 required; responses can be streamed to disk as-is.

Example (Rails / Faraday pseudo-code):

```ruby
conn = Faraday.new(url: "http://localhost:7341")
resp = conn.post("/tts", { text: "Hello from Rails" }.to_json, { "Content-Type" => "application/json" })
File.binwrite("tts.wav", resp.body)
```

## Notes

* Model weights are downloaded separately and are **not** committed
* Service loads the model once at startup and runs CPU-only
* Expect warnings from transformers / flash-attn
