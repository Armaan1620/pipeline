from typing import List

from models import Viseme


MIN_VISEME_DURATION = 0.08  # seconds

# Viseme corresponding to bilabial plosives/nasals P/B/M.
PLOSIVE_VISEME_NAME = "PBM"

# Factor by which to stretch plosives slightly.
PLOSIVE_STRETCH_FACTOR = 1.2


def _merge_adjacent_identical(visemes: List[Viseme]) -> List[Viseme]:
    """
    Merge consecutive visemes with the same name into a single span.
    """
    if not visemes:
        return []

    merged: List[Viseme] = []
    current = visemes[0]

    for v in visemes[1:]:
        assert v.start >= current.start, "Visemes must be sorted by start time."
        if v.name == current.name and abs(v.start - current.end) < 1e-6:
            # Extend current span.
            current = Viseme(
                name=current.name,
                start=current.start,
                end=max(current.end, v.end),
            )
        else:
            merged.append(current)
            current = v

    merged.append(current)
    return merged


def _enforce_min_duration(visemes: List[Viseme]) -> List[Viseme]:
    """
    Enforce a minimum viseme duration by stretching end times.

    Assumptions:
    - Visemes are sorted by start time and non-overlapping.
    - The last viseme may extend slightly beyond the true audio duration;
      callers should clip if necessary.
    """
    if not visemes:
        return []

    adjusted: List[Viseme] = []
    n = len(visemes)

    for i, v in enumerate(visemes):
        duration = v.end - v.start
        if duration >= MIN_VISEME_DURATION:
            adjusted.append(v)
            continue

        # Stretch to minimum duration.
        target_end = v.start + MIN_VISEME_DURATION

        # Do not violate ordering w.r.t next viseme.
        if i < n - 1:
            next_start = visemes[i + 1].start
            target_end = min(target_end, next_start)

        adjusted.append(
            Viseme(name=v.name, start=v.start, end=target_end)
        )

    return adjusted


def _stretch_plosives(visemes: List[Viseme]) -> List[Viseme]:
    """
    Slightly stretch plosive visemes (P, B, M grouped as PBM).

    This operates after merging and before the final merge, so that any
    small gaps introduced by stretching can be resolved by a subsequent
    merge pass if adjacent visemes share the same name.
    """
    stretched: List[Viseme] = []
    n = len(visemes)

    for i, v in enumerate(visemes):
        if v.name != PLOSIVE_VISEME_NAME:
            stretched.append(v)
            continue

        duration = v.end - v.start
        extra = duration * (PLOSIVE_STRETCH_FACTOR - 1.0)
        new_end = v.end + extra

        # Do not cross into the next viseme's start, if any.
        if i < n - 1:
            next_start = visemes[i + 1].start
            new_end = min(new_end, next_start)

        stretched.append(
            Viseme(name=v.name, start=v.start, end=new_end)
        )

    return stretched


def smooth_visemes(visemes: List[Viseme]) -> List[Viseme]:
    """
    Apply temporal smoothing to a viseme sequence.

    Steps:
    - Merge adjacent identical visemes.
    - Enforce minimum viseme duration.
    - Slightly stretch plosives (PBM).
    - Merge adjacent identical visemes again to clean up boundaries.
    """
    if not visemes:
        return []

    # Ensure sorted input; assert to fail fast if incorrect.
    visemes_sorted = sorted(visemes, key=lambda v: v.start)
    for i in range(1, len(visemes_sorted)):
        assert (
            visemes_sorted[i].start >= visemes_sorted[i - 1].start
        ), "Visemes must be sorted by start time."

    step1 = _merge_adjacent_identical(visemes_sorted)
    step2 = _enforce_min_duration(step1)
    step3 = _stretch_plosives(step2)
    step4 = _merge_adjacent_identical(step3)
    return step4

