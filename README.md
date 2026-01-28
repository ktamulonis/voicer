# voicer

Experimental local text-to-speech service using Qwen3-TTS.

## Status

- ✅ Qwen3-TTS runs locally on macOS (CPU)
- ✅ Produces WAV output via Python API
- ⚠️ Uses compatibility shims for Hugging Face internals
- ⚠️ Experimental / research-grade

## Quick test

```bash
python - <<EOF
from qwen3_tts import tts
tts("Hello from voicer")
EOF
```

## Notes

* Model weights are downloaded separately and are **not** committed
* Expect warnings from transformers / flash-attn
