# Voicer

A fully local FastAPI service for Qwen3-TTS text-to-speech and consent-based
voice cloning. It returns standard WAV files and binds to `127.0.0.1:7341` by
default, so it is intended for applications running on the same computer.

Only provide a reference recording when you own the voice or have permission
to use it.

## What you need

- Apple Silicon macOS (the service also runs on other Python-supported systems)
- Homebrew
- Python 3.11 or newer
- Enough disk space for the Qwen3-TTS 1.7B Base model

The model weights and any reference/output media are deliberately ignored by
Git. Download them locally; do not commit them to a public repository.

## Install on a clean Mac

```sh
git clone https://github.com/ktamulonis/voicer.git
cd voicer

brew install python@3.11 libsndfile ffmpeg sox
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r qwen3_tts_space/requirements.txt
```

Download the public Qwen model into the location the service expects:

```sh
python -m huggingface_hub download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir qwen3_tts_model
```

## Start and verify

```sh
source .venv/bin/activate
python server.py
```

In another terminal:

```sh
curl http://127.0.0.1:7341/health
```

Expected response:

```json
{"status":"ok"}
```

## Basic TTS

```sh
curl -X POST http://127.0.0.1:7341/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is local Qwen3 text to speech."}' \
  --output out.wav
```

## Clone a voice

Use a clean, close-mic **WAV** reference recording, ideally 5–20 seconds long,
with minimal music, reverb, or background noise. Supplying the exact transcript
of the reference significantly improves naturalness and speaker identity.

Create `reference.txt` containing exactly what is said in `reference.wav`, then
run:

```sh
curl -X POST http://127.0.0.1:7341/voice-clone \
  -F "text=Hello. This is a local cloned voice preview." \
  -F "reference=@reference.wav" \
  -F "reference_text=<reference.txt" \
  --output clone.wav
```

For an M4A or other unsupported source, make a clean mono WAV first:

```sh
ffmpeg -i source.m4a -ac 1 -ar 24000 reference.wav
```

If a clone sounds buzzy or metallic, choose a cleaner reference rather than
aggressively filtering it. The service normalizes level but preserves the
voice's full frequency range.

## API contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | none | `{"status":"ok"}` |
| `POST /tts` | JSON: `{"text":"..."}` | WAV audio |
| `POST /voice-clone` | multipart `text`, `reference`, optional `reference_text` | WAV audio |

`reference_text` is the exact transcript of the reference audio. It is optional
but strongly recommended. The clone endpoint also accepts JSON with a base64
WAV `reference` and optional `reference_text` for programmatic clients.

## Rails-friendly example

Stream the WAV response directly to a file or Active Storage attachment:

```ruby
conn = Faraday.new(url: "http://127.0.0.1:7341")
response = conn.post("/tts", { text: "Hello from Rails" }.to_json,
                     { "Content-Type" => "application/json" })
File.binwrite("tts.wav", response.body)
```

## Notes

- The service loads the model once at startup and uses CPU inference by default.
- It is research-grade local software; expect some warnings from Transformers.
- This repository vendors Qwen compatibility code in `qwen3_tts_space`; retain
  appropriate upstream attribution and licensing when redistributing it.
