from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class AudioBuffer:
    """
    Container for normalized mono float32 audio.

    Assumptions:
    - samples are 1D np.ndarray[float32] with values in [-1.0, 1.0]
    - sample_rate is in Hz and expressed as an int
    """

    samples: np.ndarray  # float32, shape (N,)
    sample_rate: int


@dataclass
class Phoneme:
    """
    Single phoneme aligned to the audio timeline.

    Time units are seconds (float) relative to the start of the audio.
    Symbols are ARPAbet (e.g. 'AH', 'M', 'P', 'S').
    """

    symbol: str
    start: float
    end: float


@dataclass
class Viseme:
    """
    Visual mouth shape corresponding to one or more phonemes.

    Time units are seconds (float) relative to the start of the audio.
    """

    name: str
    start: float
    end: float


PhonemeList = List[Phoneme]
VisemeList = List[Viseme]

