import math
import os
import subprocess
import tempfile
import wave
from typing import List

import numpy as np

from models import AudioBuffer


def _audio_to_int16_pcm(audio: AudioBuffer) -> bytes:
    """
    Convert normalized float32 mono audio in [-1.0, 1.0] to int16 PCM.
    """
    samples = audio.samples
    assert samples.dtype == np.float32, "Audio must be float32."
    assert samples.ndim == 1, "Audio must be mono (1D)."

    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    return int16.tobytes()


def mux_frames_and_audio_to_mp4(
    frames: List[np.ndarray],
    audio: AudioBuffer,
    output_path: str,
    fps: int = 8,
) -> None:
    """
    Mux rendered RGB frames and mono audio into an MP4 file using ffmpeg.

    Strategy:
    - Video is streamed via stdin (raw RGB).
    - Audio is written once to a temporary WAV file.
    - ffmpeg handles synchronization deterministically.

    This avoids FIFO deadlocks and pipe backpressure issues.
    """
    assert fps == 8, "FPS must be exactly 8."
    assert len(frames) > 0, "At least one frame is required."
    assert audio.sample_rate == 22050, "Audio sample rate must be 22050 Hz."

    # Validate frames
    h, w, c = frames[0].shape
    assert c == 3, "Frames must be RGB."
    for i, frame in enumerate(frames):
        assert frame.shape == (h, w, 3), f"Frame {i} shape mismatch."
        assert frame.dtype == np.uint8, "Frames must be uint8."

    # Convert audio to int16 PCM
    audio_pcm = _audio_to_int16_pcm(audio)

    # Create temp directory for ffmpeg artifacts
    tmpdir = tempfile.mkdtemp(prefix="ffmpeg-")
    audio_wav_path = os.path.join(tmpdir, "audio.wav")

    # Write temporary WAV file (ffmpeg-safe)
    with wave.open(audio_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(audio.sample_rate)
        wf.writeframes(audio_pcm)

    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",          # video from stdin
        "-i",
        audio_wav_path,    # audio from temp WAV
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]

    # Start ffmpeg
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Stream video frames
    try:
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except Exception:
        proc.kill()
        raise

    # Wait for ffmpeg
    proc.wait()

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}):\n"
            f"{stderr.decode('utf-8', errors='ignore')}"
        )

    # Final sanity check
    audio_duration = audio.samples.shape[0] / float(audio.sample_rate)
    expected_frames = math.ceil(audio_duration * fps)
    assert len(frames) == expected_frames, (
        "Frame count must match ceil(audio_duration * FPS)."
    )

    # Cleanup temp files
    try:
        os.remove(audio_wav_path)
        os.rmdir(tmpdir)
    except OSError:
        pass
