import math
import os
import sys
from typing import List

import numpy as np

from alignment.phonemes import extract_phonemes
from audio.ingest import load_audio_from_wav_bytes
from models import AudioBuffer, Viseme
from mux.ffmpeg import mux_frames_and_audio_to_mp4
from render.renderer import FPS, SpriteRenderer
from visemes.map import phonemes_to_visemes
from visemes.smooth import smooth_visemes


def _read_file_bytes(path: str) -> bytes:
    """Read an entire file as bytes. Only used for input audio (allowed)."""
    with open(path, "rb") as f:
        return f.read()


def _orchestrate(
    audio_buffer: AudioBuffer,
    sprites_dir: str,
    output_path: str,
) -> None:
    """
    Wire the full pipeline end-to-end for a single audio input.

    Steps:
    - Ingested AudioBuffer -> Phonemes (forced alignment, external).
    - Phonemes -> Visemes (static mapping).
    - Visemes -> Smoothed visemes (temporal cleanup).
    - Smoothed visemes -> Frames (sprite renderer at 8 FPS).
    - Frames + audio -> MP4 via ffmpeg muxing.
    """
    # STEP 2 — Phoneme extraction (forced alignment)
    transcript = "ah"

    phonemes = extract_phonemes(audio_buffer , transcript)
    # Output must be sorted; assert to fail fast if aligner returns bad data.
    for i in range(1, len(phonemes)):
        assert (
            phonemes[i].start >= phonemes[i - 1].start
        ), "Phonemes must be sorted by start time."

    print(f"Extracted {len(phonemes)} phonemes")

    # STEP 3 — Phoneme → Viseme mapping
    visemes_raw = phonemes_to_visemes(phonemes)

    # STEP 4 — Viseme smoothing
    visemes_smoothed: List[Viseme] = smooth_visemes(visemes_raw)

    print(f"Generated {len(visemes_smoothed)} visemes after smoothing")

    # Determine total duration from audio.
    duration = audio_buffer.samples.shape[0] / float(audio_buffer.sample_rate)
    assert duration > 0.0, "Audio duration must be positive."

    # Sanity: ensure last viseme does not start after audio end.
    if visemes_smoothed:
        assert (
            visemes_smoothed[-1].start <= duration
        ), "Last viseme starts after audio ends."

    # STEP 5 — Frame rendering
    unique_viseme_names = sorted({v.name for v in visemes_smoothed}) or ["REST"]
    renderer = SpriteRenderer(
        sprites_dir=sprites_dir,
        viseme_names=unique_viseme_names,
    )

    frames = renderer.render_sequence(visemes_smoothed, duration=duration)

    print(f"Rendered {len(frames)} frames at {FPS} FPS")

    # STEP 6 — Muxing audio + frames
    mux_frames_and_audio_to_mp4(
        frames=frames,
        audio=audio_buffer,
        output_path=output_path,
        fps=FPS,
    )

    # STEP 7 — Sanity checks after muxing.
    expected_frame_count = math.ceil(duration * FPS)
    assert (
        len(frames) == expected_frame_count
    ), "Frame count must be ceil(duration * FPS)."


def main(argv: list[str]) -> int:
    """
    Basic command-line entry point.

    Usage:
        python -m main <input_wav_path> <sprites_dir> <output_mp4_path>

    Notes:
    - Only the final MP4 file is written to disk.
    - Phoneme extraction is not implemented by default and will raise
      NotImplementedError until a concrete forced aligner is integrated.
    """
    if len(argv) != 4:
        print(
            "Usage: python -m main <input_wav_path> <sprites_dir> <output_mp4_path>",
            file=sys.stderr,
        )
        return 1

    _, input_wav, sprites_dir, output_mp4 = argv

    assert os.path.isfile(input_wav), f"Input WAV not found: {input_wav}"
    assert os.path.isdir(
        sprites_dir
    ), f"Sprite directory does not exist: {sprites_dir}"

    wav_bytes = _read_file_bytes(input_wav)

    # STEP 1 — Audio ingestion
    audio_buffer = load_audio_from_wav_bytes(wav_bytes)

    # Orchestrate the complete pipeline.
    _orchestrate(audio_buffer, sprites_dir=sprites_dir, output_path=output_mp4)

    # If we reach this point without assertion failures or exceptions,
    # the pipeline has produced a deterministic MP4 video.
    audio_duration = audio_buffer.samples.shape[0] / float(
        audio_buffer.sample_rate
    )
    expected_frames = math.ceil(audio_duration * FPS)
    print(
        f"Done. Audio duration ~{audio_duration:.3f}s, "
        f"expected frames = {expected_frames}, FPS = {FPS}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

