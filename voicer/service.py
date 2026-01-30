import base64
import io
import sys
import threading
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
SPACE_DIR = BASE_DIR / "qwen3_tts_space"
MODEL_DIR = BASE_DIR / "qwen3_tts_model"

# Make qwen_tts importable without changing global env
if str(SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(SPACE_DIR))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402


class QwenTTSService:
    """Thread-safe wrapper around Qwen3-TTS for local inference."""

    def __init__(self, device: str = "cpu") -> None:
        warnings.filterwarnings("ignore", message=".*flash_attn.*")
        warnings.filterwarnings("ignore", message=".*xformers.*")
        warnings.filterwarnings("ignore", message=".*Some weights of.*")

        self.device = device
        self._lock = threading.Lock()
        self.model = Qwen3TTSModel.from_pretrained(str(MODEL_DIR), device=self.device)

    # ------------------------
    # Public APIs
    # ------------------------

    def synthesize_to_wav_bytes(self, text: str) -> Tuple[bytes, int]:
        with self._lock:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=None,
            )
        wav = wavs[0]
        return self._wav_to_bytes(wav, sr), sr

    def voice_clone_to_wav_bytes(
        self,
        text: str,
        ref_audio_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Tuple[bytes, int]:
        # Decode + clean reference audio
        ref_wav, ref_sr = self._read_audio_bytes(ref_audio_bytes)
        ref_wav = self._clean_reference_audio(ref_wav, ref_sr)

        with self._lock:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                ref_audio=(ref_wav, ref_sr),
                x_vector_only_mode=False,
                speaker_strength=0.7,  # 🔑 reduces robotic tone
            )

        wav = wavs[0]
        return self._wav_to_bytes(wav, sr), sr

    # ------------------------
    # Audio helpers
    # ------------------------

    @staticmethod
    def _wav_to_bytes(wav, sr: int) -> bytes:
        with io.BytesIO() as buffer:
            sf.write(buffer, wav, sr, format="WAV")
            return buffer.getvalue()

    @staticmethod
    def _read_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
        with io.BytesIO(data) as buffer:
            audio, sr = sf.read(buffer, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    @staticmethod
    def _clean_reference_audio(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Fixes buzzing + robotic tone by:
        - removing DC / low-frequency rumble
        - normalizing loudness
        - padding silence at start
        """

        # High-pass (simple 1st-order filter at ~80Hz)
        hp_cut = int(sr * 0.003)
        if hp_cut > 1:
            audio = audio - np.convolve(audio, np.ones(hp_cut) / hp_cut, mode="same")

        # Normalize RMS
        rms = np.sqrt(np.mean(audio**2) + 1e-9)
        audio = audio / max(rms, 0.02)

        # Add 200ms silence pad
        pad = np.zeros(int(0.2 * sr), dtype=np.float32)
        audio = np.concatenate([pad, audio])

        return audio


# ------------------------
# Singleton
# ------------------------

_SERVICE: Optional[QwenTTSService] = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> QwenTTSService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = QwenTTSService(device="cpu")
    return _SERVICE


def decode_base64_audio(data: str) -> bytes:
    if data.startswith("data:audio") and "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)

