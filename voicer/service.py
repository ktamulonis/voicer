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

try:  # Optional: better IIR filters if scipy is installed
    from scipy.signal import butter, sosfiltfilt

    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

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
        wav = self._postprocess_output_audio(wavs[0], sr)
        return self._wav_to_bytes(wav, sr), sr

    def voice_clone_to_wav_bytes(
        self,
        text: str,
        ref_audio_bytes: bytes,
        filename: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> Tuple[bytes, int]:
        # Preserve the reference's full vocal range. Earlier versions used an
        # 8 kHz low-pass plus output clipping, which caused a thin, metallic
        # "tin can" quality for otherwise clean recordings.
        ref_wav, ref_sr = self._read_audio_bytes(ref_audio_bytes)
        ref_wav = self._preprocess_reference_audio(ref_wav, ref_sr)

        max_ref_seconds = 20
        max_len = int(ref_sr * max_ref_seconds)
        if ref_wav.shape[0] > max_len:
            # Limit prosody transfer by trimming to the first 20s.
            ref_wav = ref_wav[:max_len]

        with self._lock:
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                ref_audio=(ref_wav, ref_sr),
                # Supplying the exact transcript enables Qwen's in-context
                # learning mode, retaining far more natural timbre and prosody.
                ref_text=reference_text.strip() if reference_text else None,
                x_vector_only_mode=not bool(reference_text and reference_text.strip()),
            )

        wav = self._postprocess_output_audio(wavs[0], sr)
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

    def _preprocess_reference_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Normalize RMS to -20 dBFS
        target_rms = 10 ** (-20 / 20)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
        if rms > 1e-6:
            scale = min(target_rms / rms, 10.0)
            audio = audio * scale

        return audio.astype(np.float32)

    def _postprocess_output_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Keep the model's output intact; rescale only if it would clip.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.99:
            audio = audio * (0.99 / peak)
        return audio.astype(np.float32)

    def _band_limit(self, audio: np.ndarray, sr: int, low_cut: float, high_cut: float) -> np.ndarray:
        if _SCIPY_AVAILABLE:
            nyq = 0.5 * sr
            low = max(low_cut / nyq, 1e-5)
            high = min(high_cut / nyq, 0.999)
            if high <= low:
                return audio
            sos = butter(2, [low, high], btype="band", output="sos")
            return sosfiltfilt(sos, audio).astype(np.float32)

        return self._fft_band_limit(audio, sr, low_cut, high_cut)

    def _low_pass(self, audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        if _SCIPY_AVAILABLE:
            nyq = 0.5 * sr
            high = min(cutoff / nyq, 0.999)
            sos = butter(2, high, btype="low", output="sos")
            return sosfiltfilt(sos, audio).astype(np.float32)

        return self._fft_low_pass(audio, sr, cutoff)

    @staticmethod
    def _fft_band_limit(audio: np.ndarray, sr: int, low_cut: float, high_cut: float) -> np.ndarray:
        n = int(audio.shape[0])
        if n == 0:
            return audio
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        low_transition = max(10.0, low_cut * 0.2)
        high_transition = max(200.0, high_cut * 0.2)

        gain = np.ones_like(freqs, dtype=np.float32)

        # High-pass ramp
        if low_cut > 0:
            low_start = max(low_cut - low_transition, 0.0)
            low_end = low_cut + low_transition
            below = freqs <= low_start
            above = freqs >= low_end
            mid = (~below) & (~above)
            gain[below] = 0.0
            t = (freqs[mid] - low_start) / max(low_end - low_start, 1e-6)
            gain[mid] = 0.5 - 0.5 * np.cos(np.pi * t)

        # Low-pass ramp
        if high_cut > 0:
            high_start = max(high_cut - high_transition, 0.0)
            high_end = high_cut + high_transition
            below = freqs <= high_start
            above = freqs >= high_end
            mid = (~below) & (~above)
            gain[above] = 0.0
            t = (freqs[mid] - high_start) / max(high_end - high_start, 1e-6)
            gain[mid] = gain[mid] * (0.5 + 0.5 * np.cos(np.pi * t))

        fft *= gain
        out = np.fft.irfft(fft, n=n)
        return out.astype(np.float32)

    @staticmethod
    def _fft_low_pass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        n = int(audio.shape[0])
        if n == 0:
            return audio
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        transition = max(200.0, cutoff * 0.2)
        start = max(cutoff - transition, 0.0)
        end = cutoff + transition

        gain = np.ones_like(freqs, dtype=np.float32)
        below = freqs <= start
        above = freqs >= end
        mid = (~below) & (~above)
        gain[above] = 0.0
        t = (freqs[mid] - start) / max(end - start, 1e-6)
        gain[mid] = 0.5 + 0.5 * np.cos(np.pi * t)

        fft *= gain
        out = np.fft.irfft(fft, n=n)
        return out.astype(np.float32)

    @staticmethod
    def _soft_clip(audio: np.ndarray, drive: float = 1.2) -> np.ndarray:
        drive = max(drive, 1.0)
        return np.tanh(audio * drive) / np.tanh(drive)


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
