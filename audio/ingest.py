import io
import wave
from typing import Union

import numpy as np

from models import AudioBuffer


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    """
    Normalize audio to have max abs value of 1.0.

    Assumptions:
    - samples is a 1D float32 numpy array.
    - In-place modification is avoided to keep behavior explicit.
    """
    assert samples.ndim == 1, "Audio samples must be mono (1D)."
    assert samples.dtype == np.float32, "Audio samples must be float32."

    max_abs = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    if max_abs > 0.0:
        samples = samples / max_abs
    return samples


def load_audio_from_wav_bytes(data: Union[bytes, bytearray, memoryview]) -> AudioBuffer:
    """
    Ingest a mono float32 WAV buffer held in memory.

    Constraints enforced:
    - dtype == float32
    - mono
    - sample_rate == 22050 Hz
    - All time units are seconds in downstream components.

    No disk I/O is performed here. The caller is responsible for
    providing the WAV contents as an in-memory buffer.
    """
    assert isinstance(
        data, (bytes, bytearray, memoryview)
    ), "Expected in-memory WAV buffer (bytes-like object)."

    with io.BytesIO(data) as bio:
        with wave.open(bio, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            # Validate basic format assumptions before reading samples.
            assert n_channels == 1, "Audio must be mono."
            # 4-byte samples imply 32-bit PCM. We interpret them as IEEE float32.
            assert (
                sampwidth == 4
            ), f"Expected 32-bit float samples (4 bytes), got {sampwidth} bytes."
            assert (
                sample_rate == 22050
            ), f"Expected sample rate 22050 Hz, got {sample_rate}."

            raw_frames = wf.readframes(n_frames)

    # Interpret raw PCM as little-endian float32.
    samples = np.frombuffer(raw_frames, dtype="<f4").astype(
        np.float32, copy=False
    )

    # Normalize once at ingestion time.
    samples = _normalize_audio(samples)

    return AudioBuffer(samples=samples, sample_rate=sample_rate)

