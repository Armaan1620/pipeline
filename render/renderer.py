from pathlib import Path
import random
from typing import Dict, List

import numpy as np
from PIL import Image

FPS = 8


class SpriteRenderer:
    def __init__(
        self,
        sprites_dir: str,
        viseme_names: List[str],
        seed: int = 0,
    ) -> None:
        self.sprites_dir = Path(sprites_dir)
        self.rng = random.Random(seed)

        # 1️⃣ Load base image FIRST (no resizing)
        self.base_image = self._load_base_sprite(
            ["REST.png", "base.png", "neutral.png"]
        )

        # 2️⃣ Load viseme sprites (resized to base)
        self.viseme_sprites: Dict[str, Image.Image] = {}
        for name in viseme_names:
            path = self.sprites_dir / f"{name}.png"
            if path.exists():
                self.viseme_sprites[name] = self._load_dependent_sprite(path)

        # 3️⃣ Optional blink sprite
        blink_path = self.sprites_dir / "blink.png"
        self.blink_sprite = (
            self._load_dependent_sprite(blink_path)
            if blink_path.exists()
            else None
        )

    # ---------- sprite loading helpers ----------

    def _load_base_sprite(self, candidates: List[str]) -> Image.Image:
        """
        Load the base sprite WITHOUT resizing.
        This defines the canonical frame size.
        """
        for name in candidates:
            path = self.sprites_dir / name
            if path.exists():
                return Image.open(path).convert("RGBA")

        raise FileNotFoundError(
            f"No base sprite found. Expected one of: {candidates}"
        )

    def _load_dependent_sprite(self, path: Path) -> Image.Image:
        """
        Load a sprite and resize it to match base_image.
        """
        img = Image.open(path).convert("RGBA")
        img = img.resize(self.base_image.size, Image.NEAREST)
        return img

    # ---------- rendering ----------

    def render_sequence(self, visemes, duration: float) -> List[np.ndarray]:
        frame_count = int(np.ceil(duration * FPS))
        assert frame_count > 0

        frames: List[np.ndarray] = []

        for frame_idx in range(frame_count):
            t = frame_idx / FPS
            frame = self._render_frame(t, visemes)
            frames.append(frame)

        return frames

    def _render_frame(self, t, visemes) -> np.ndarray:
        # Start from base
        frame = self.base_image.copy()

        # Find active viseme
        active = None
        for v in visemes:
            if v.start <= t < v.end:
                active = v.name
                break

        if active and active in self.viseme_sprites:
            frame.alpha_composite(self.viseme_sprites[active])

        # Blink (optional)
        if self.blink_sprite and self._blink_active(t):
            frame.alpha_composite(self.blink_sprite)

        # Convert to RGB numpy array
        rgb = frame.convert("RGB")
        return np.array(rgb, dtype=np.uint8)

    def _blink_active(self, t: float) -> bool:
        # Simple deterministic blink every ~3 seconds
        return int(t * 10) % 30 == 0
