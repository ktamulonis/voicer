import sys
from pathlib import Path
import torch
import soundfile as sf

BASE_DIR = Path(__file__).parent
SPACE_DIR = BASE_DIR / "qwen3_tts_space"
MODEL_DIR = BASE_DIR / "qwen3_tts_model"

# Make qwen_tts importable
sys.path.insert(0, str(SPACE_DIR))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model (this registers qwen3_tts with Transformers internally)
tts_model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR),
    device=DEVICE
)

def tts(text: str, output_path="out.wav"):
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        voice_clone_prompt=None
    )
    sf.write(output_path, wavs[0], sr)
    return output_path
