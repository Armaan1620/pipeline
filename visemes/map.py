from typing import Dict, List

from models import Phoneme, Viseme


# Static ARPAbet → viseme name mapping.
# This collapses many phonemes into ~10 visual mouth shapes.
#
# Assumptions:
# - Symbols are uppercase ARPAbet without stress markers (e.g. 'AH', not 'AH0').
# - Unknown symbols fall back to a neutral/closed viseme.
PHONEME_TO_VISEME: Dict[str, str] = {
    # Closed lips (bilabial plosives / nasals)
    "P": "PBM",
    "B": "PBM",
    "M": "PBM",
    # Labiodental
    "F": "FV",
    "V": "FV",
    # Dentals / alveolars with teeth showing
    "TH": "TH",
    "DH": "TH",
    # Alveolar fricatives
    "S": "SZ",
    "Z": "SZ",
    # Affricates
    "CH": "CHJ",
    "JH": "CHJ",
    # Velar / palatal
    "K": "KNG",
    "G": "KNG",
    "NG": "KNG",
    "Y": "Y",
    # Rounded vowels / w
    "UW": "UW",
    "UH": "UW",
    "OW": "UW",
    "W": "UW",
    # Wide open vowels
    "AA": "AA",
    "AE": "AA",
    "AH": "AA",
    "AO": "AA",
    # Front vowels
    "IY": "IY",
    "IH": "IY",
    "EY": "IY",
    "EH": "IY",
    # Rounded mid vowels
    "ER": "ER",
    "R": "ER",
    # Liquids / laterals
    "L": "L",
    # Glottal / neutral-ish
    "HH": "REST",
    # Silence / pause
    "SIL": "REST",
}


def _normalize_symbol(symbol: str) -> str:
    """
    Strip common ARPAbet stress markers (0/1/2) and normalize to uppercase.
    """
    s = symbol.strip().upper()
    if s and s[-1] in {"0", "1", "2"}:
        s = s[:-1]
    return s


def phonemes_to_visemes(phonemes: List[Phoneme]) -> List[Viseme]:
    """
    Convert a list of aligned phonemes into a list of visemes.

    Behavior:
    - Each phoneme is mapped to a viseme name via `PHONEME_TO_VISEME`.
    - Unknown phonemes map to 'REST' to avoid undefined states.
    - Output visemes preserve the original timing of the source phonemes.
    - No temporal smoothing is performed here; that is handled separately.
    """
    visemes: List[Viseme] = []

    for p in phonemes:
        assert p.end >= p.start, "Phoneme end time must be >= start time."
        symbol = _normalize_symbol(p.symbol)
        viseme_name = PHONEME_TO_VISEME.get(symbol, "REST")
        visemes.append(Viseme(name=viseme_name, start=p.start, end=p.end))

    # Input is expected to be sorted; keep order unchanged here.
    return visemes

