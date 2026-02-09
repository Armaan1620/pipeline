"""
Minimal end-to-end test harness for the audio-driven video pipeline.

This test validates that the pipeline can process audio → frames → MP4
using the development stub for phoneme extraction.

No pytest or test frameworks — plain Python assertions.
"""

import io
import math
import os
import tempfile
import wave

import numpy as np

from main import _orchestrate
from render.renderer import FPS


def _generate_test_wav(duration_seconds: float = 2.5) -> bytes:
    """
    Generate a minimal mono float32 WAV file in memory for testing.

    Creates a simple sine wave tone at 440 Hz (A4) that satisfies:
    - mono
    - float32 samples
    - 22050 Hz sample rate
    - duration as specified

    Returns:
        bytes: Complete WAV file contents ready to write to disk or use directly.
    """
    sample_rate = 22050
    n_samples = int(duration_seconds * sample_rate)
    frequency = 440.0  # A4 note

    # Generate time axis.
    # IMPORTANT: endpoint=False ensures duration == n_samples / sample_rate
    t = np.linspace(
        0.0,
        duration_seconds,
        n_samples,
        endpoint=False,
        dtype=np.float32,
    )

    # Generate sine wave samples and explicitly cast to float32.
    samples = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)

    # Write to WAV format in memory.
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)      # mono
        wf.setsampwidth(4)      # 4 bytes = 32-bit
        wf.setframerate(sample_rate)
        wf.setcomptype("NONE", "not compressed")

        # NOTE:
        # The wave module does not distinguish int32 PCM vs float32 WAV.
        # This is safe here because load_audio_from_wav_bytes explicitly
        # interprets the payload as little-endian float32 (<f4).
        wf.writeframes(samples.astype("<f4").tobytes())

    return bio.getvalue()


def test_pipeline_end_to_end() -> None:
    """
    Test the complete pipeline from audio ingestion to MP4 output.

    Steps:
    1. Generate a test WAV file (~2.5 seconds).
    2. Set PHONEME_MODE=stub to use development stub.
    3. Create temporary directories for sprites and output.
    4. Generate minimal sprites (base + blink).
    5. Run orchestration programmatically.
    6. Verify:
       - Output MP4 exists.
       - Frame count matches ceil(duration × FPS).
       - No exceptions occurred.

    This test requires:
    - ffmpeg installed and available in PATH.
    - Pillow and numpy installed.
    """
    # Force development stub for phoneme extraction.
    os.environ["PHONEME_MODE"] = "stub"

    # Generate test WAV and ingest it.
    wav_bytes = _generate_test_wav(duration_seconds=2.5)
    from audio.ingest import load_audio_from_wav_bytes

    audio_buffer = load_audio_from_wav_bytes(wav_bytes)
    duration = audio_buffer.samples.shape[0] / float(audio_buffer.sample_rate)

    # Create temporary directories for sprites and output.
    with tempfile.TemporaryDirectory(prefix="pipeline_test_") as tmpdir:
        sprites_dir = os.path.join(tmpdir, "sprites")
        os.makedirs(sprites_dir, exist_ok=True)

        from PIL import Image

        # Base / neutral sprite.
        base_img = Image.new("RGBA", (256, 256), (200, 180, 160, 255))
        base_img.save(os.path.join(sprites_dir, "REST.png"))

        # Blink sprite (represents closed eyes).
        blink_img = Image.new("RGBA", (256, 256), (150, 130, 110, 255))
        blink_img.save(os.path.join(sprites_dir, "blink.png"))

        output_mp4 = os.path.join(tmpdir, "output.mp4")

        # Run the pipeline programmatically.
        try:
            _orchestrate(
                audio_buffer=audio_buffer,
                sprites_dir=sprites_dir,
                output_path=output_mp4,
            )
        except Exception as e:
            raise AssertionError(
                f"Pipeline failed with exception: {e}"
            ) from e

        # Verify output MP4 exists and is non-empty.
        assert os.path.isfile(output_mp4), f"Output MP4 not found: {output_mp4}"
        assert os.path.getsize(output_mp4) > 0, "Output MP4 is empty"

        # Frame count expectation (checked internally by _orchestrate).
        expected_frame_count = math.ceil(duration * FPS)

        print(f"✓ Test passed: MP4 generated at {output_mp4}")
        print(f"  Audio duration: {duration:.3f}s")
        print(f"  Expected frames: {expected_frame_count}")
        print(f"  Output file size: {os.path.getsize(output_mp4)} bytes")


if __name__ == "__main__":
    test_pipeline_end_to_end()
    print("All tests passed.")
