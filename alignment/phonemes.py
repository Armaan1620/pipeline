import os
from typing import List

from models import AudioBuffer, Phoneme


def _extract_phonemes_dev_stub(audio: AudioBuffer) -> List[Phoneme]:
    """
    Deterministic development stub for phoneme extraction (testing only).

    Returns a hard-coded sequence of phonemes covering ~2 seconds that exercises
    multiple visemes. This allows testing the pipeline end-to-end without a
    forced aligner installed.

    Phonemes chosen to exercise different viseme groups:
    - AA (wide open vowel) -> AA viseme
    - M (bilabial nasal) -> PBM viseme
    - F (labiodental) -> FV viseme
    - S (alveolar fricative) -> SZ viseme
    - AH (open vowel) -> AA viseme
    - P (bilabial plosive) -> PBM viseme

    Timestamps:
    - Start at 0.0
    - Sorted and non-overlapping
    - Total duration ~2 seconds (will be clipped to actual audio duration)
    """
    # Hard-coded phoneme sequence with deterministic timestamps.
    stub_phonemes = [
        Phoneme(symbol="AA", start=0.0, end=0.25),
        Phoneme(symbol="M", start=0.25, end=0.45),
        Phoneme(symbol="F", start=0.45, end=0.65),
        Phoneme(symbol="S", start=0.65, end=0.85),
        Phoneme(symbol="AH", start=0.85, end=1.10),
        Phoneme(symbol="P", start=1.10, end=1.30),
        Phoneme(symbol="AA", start=1.30, end=1.55),
        Phoneme(symbol="M", start=1.55, end=1.75),
        Phoneme(symbol="SIL", start=1.75, end=2.00),
    ]

    # Clip to actual audio duration if stub sequence is longer.
    audio_duration = audio.samples.shape[0] / float(audio.sample_rate)
    clipped = [
        Phoneme(symbol=p.symbol, start=p.start, end=min(p.end, audio_duration))
        for p in stub_phonemes
        if p.start < audio_duration
    ]

    return clipped


def _extract_phonemes_with_mfa(
    audio: AudioBuffer,
    transcript: str,
    mfa_model_path: str,
) -> List[Phoneme]:
    """
    Extract phonemes using Montreal Forced Aligner (MFA).

    This function defines the API shape for MFA integration but does not yet
    implement the actual execution. The expected workflow is:

    1. Write audio buffer to a temporary WAV file (mono, float32, 22050 Hz).
    2. Write transcript to a temporary text file.
    3. Invoke MFA via subprocess with:
       - MFA model path (e.g., "english_us_arpa" or custom model directory).
       - Input WAV and transcript paths.
       - Output directory for alignment results.
    4. Parse MFA's phoneme-level output (typically TextGrid or similar format).
    5. Convert parsed alignments to List[Phoneme] with ARPAbet symbols and
       timestamps in seconds.
    6. Clean up temporary files.
    7. Return sorted, non-overlapping phonemes.

    To preserve determinism:
    - Use fixed MFA model version.
    - Disable any randomization in MFA configuration.
    - Parse output format consistently (handle edge cases explicitly).

    Args:
        audio: Normalized mono float32 AudioBuffer at 22050 Hz.
        transcript: Text transcript corresponding to the audio (required by MFA).
        mfa_model_path: Path to MFA acoustic model or model name.

    Returns:
        List[Phoneme] sorted by start time, non-overlapping, with ARPAbet symbols.

    Raises:
        NotImplementedError: This function is a scaffold and not yet implemented.
    """
    raise NotImplementedError(
        "MFA integration is not yet implemented. "
        "This function defines the API shape for future MFA integration. "
        "To use the development stub for testing, set PHONEME_MODE=stub."
    )


def extract_phonemes(audio: AudioBuffer) -> List[Phoneme]:
    """
    Perform forced alignment to extract ARPAbet phonemes with timestamps.

    Modes:
    - DEV STUB (PHONEME_MODE=stub): Returns a deterministic hard-coded phoneme
      sequence for testing. Only activated when environment variable is explicitly
      set to "stub". Does NOT silently fall back to stub mode.
    - REAL ALIGNER (default): Raises NotImplementedError until a forced aligner
      (e.g. MFA) is integrated via `_extract_phonemes_with_mfa`.

    Assumptions:
    - `audio` has already been validated and normalized.
    - Time units are seconds (float) relative to audio start.
    - Output is sorted by `start` and non-overlapping.

    To use the development stub for testing:
        PHONEME_MODE=stub python main.py input.wav render/sprites output.mp4

    To integrate MFA:
        Implement `_extract_phonemes_with_mfa` and call it from here with
        appropriate transcript and model path arguments.
    """
    phoneme_mode = os.environ.get("PHONEME_MODE", "").strip().lower()

    if phoneme_mode == "stub":
        return _extract_phonemes_dev_stub(audio)

    # Default behavior: fail fast until real aligner is integrated.
    raise NotImplementedError(
        "Phoneme extraction is not implemented. "
        "Set PHONEME_MODE=stub for development/testing, or "
        "integrate MFA via `_extract_phonemes_with_mfa`."
    )

