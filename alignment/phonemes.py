import os
import subprocess
import tempfile
from typing import List

from praatio import textgrid

from models import AudioBuffer, Phoneme


# ---------------------------------------------------------------------
# DEV STUB
# ---------------------------------------------------------------------

def _extract_phonemes_dev_stub(audio: AudioBuffer) -> List[Phoneme]:
    duration = audio.samples.shape[0] / audio.sample_rate

    phonemes = [
        ("SIL", 0.00, 0.20),
        ("AA", 0.20, 0.45),
        ("M", 0.45, 0.65),
        ("F", 0.65, 0.85),
        ("S", 0.85, 1.05),
        ("AH", 1.05, 1.30),
        ("P", 1.30, 1.55),
        ("AA", 1.55, 1.85),
        ("SIL", 1.85, 2.10),
    ]

    out: List[Phoneme] = []
    for symbol, start, end in phonemes:
        if start >= duration:
            break
        out.append(
            Phoneme(
                symbol,
                start,
                min(end, duration),
            )
        )

    return out


# ---------------------------------------------------------------------
# REAL MFA EXTRACTION
# ---------------------------------------------------------------------

def _extract_phonemes_with_mfa(
    audio: AudioBuffer,
    transcript: str,
) -> List[Phoneme]:

    with tempfile.TemporaryDirectory(prefix="mfa_run_") as work_dir:
        corpus_dir = os.path.join(work_dir, "corpus")
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(corpus_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        wav_path = os.path.join(corpus_dir, "utt.wav")
        txt_path = os.path.join(corpus_dir, "utt.txt")

        from audio.ingest import write_wav_file

        write_wav_file(wav_path, audio)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.strip())

        cmd = [
            "mfa",
            "align",
            corpus_dir,
            "english_us_arpa",
            "english_mfa",
            output_dir,
            "--clean",
            "--beam",
            "100",
            "--retry_beam",
            "400",
            "--single_speaker",
            "--quiet",
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"MFA alignment failed:\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

        textgrid_path = None
        for root, _, files in os.walk(output_dir):
            for fn in files:
                if fn.lower().endswith(".textgrid"):
                    textgrid_path = os.path.join(root, fn)
                    break

        if textgrid_path is None:
            raise RuntimeError("MFA completed but no TextGrid was produced.")

        tg = textgrid.openTextgrid(
            textgrid_path,
            includeEmptyIntervals=False,
        )

        if "phones" not in tg.tierNames:
            raise RuntimeError(
                f"TextGrid missing 'phones' tier. Found: {tg.tierNames}"
            )

        phones_tier = tg.getTier("phones")

        phonemes: List[Phoneme] = []
        for start, end, symbol in phones_tier.entries:
            if not symbol:
                continue
            phonemes.append(
                Phoneme(
                    symbol.upper(),
                    float(start),
                    float(end),
                )
            )

        if not phonemes:
            raise RuntimeError("No phonemes extracted from MFA output.")

        return phonemes


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def extract_phonemes(
    audio: AudioBuffer,
    transcript: str | None = None,
) -> List[Phoneme]:

    mode = os.environ.get("PHONEME_MODE")

    if mode == "stub":
        return _extract_phonemes_dev_stub(audio)

    if mode != "real":
        raise RuntimeError(
            "PHONEME_MODE must be explicitly set to 'stub' or 'real'."
        )

    if not transcript:
        raise RuntimeError(
            "Transcript is required when PHONEME_MODE=real."
        )

    return _extract_phonemes_with_mfa(audio, transcript)
