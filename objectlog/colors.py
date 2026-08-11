"""Naming the dominant colour of a patch of pixels.

The detector tells us *what* something is. This tells us what colour it is, so
the log can say "Person wearing blue" instead of just "person". It is honest
about uncertainty: a washed-out or busy patch returns None rather than a guess.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# (name, hue_low, hue_high) in degrees. Red wraps, so it appears twice.
_HUE_BANDS = [
    ("red", 0, 12),
    ("orange", 12, 40),
    ("yellow", 40, 65),
    ("green", 65, 160),
    ("teal", 160, 195),
    ("blue", 195, 255),
    ("purple", 255, 295),
    ("pink", 295, 340),
    ("red", 340, 360),
]


def _rgb_to_hsv(pixels: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV. Input (N,3) uint8, output (N,3) float.

    Hue in degrees 0-360, saturation and value in 0-1.
    """
    rgb = pixels.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    # Which channel is the max decides which 60-degree sector we are in.
    is_r = nonzero & (maxc == r)
    is_g = nonzero & (maxc == g) & ~is_r
    is_b = nonzero & (maxc == b) & ~is_r & ~is_g
    hue[is_r] = ((g[is_r] - b[is_r]) / delta[is_r]) % 6.0
    hue[is_g] = ((b[is_g] - r[is_g]) / delta[is_g]) + 2.0
    hue[is_b] = ((r[is_b] - g[is_b]) / delta[is_b]) + 4.0
    hue *= 60.0

    sat = np.zeros_like(maxc)
    sat[maxc > 1e-6] = delta[maxc > 1e-6] / maxc[maxc > 1e-6]
    return np.stack([hue, sat, maxc], axis=1)


def analyse(patch: np.ndarray, min_pixels: int = 40
            ) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    """Name a patch's dominant colour and give a representative RGB for it.

    Returns (name, rgb). The RGB is the median of the pixels that actually
    won -- not the mean of the whole patch, which would be muddied by
    background and shadow and would not match the name.

    `patch` is an (H, W, 3) uint8 RGB array.
    """
    if patch is None or patch.size == 0:
        return None, None
    flat = patch.reshape(-1, 3)
    if flat.shape[0] < min_pixels:
        return None, None

    # Sample rather than process every pixel -- this runs per object per frame
    # on a Pi, and 4000 pixels is plenty to find a dominant colour.
    if flat.shape[0] > 4000:
        idx = np.linspace(0, flat.shape[0] - 1, 4000).astype(np.int32)
        flat = flat[idx]

    hsv = _rgb_to_hsv(flat)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    def median_rgb(mask) -> Optional[Tuple[int, int, int]]:
        selected = flat[mask]
        if selected.shape[0] == 0:
            return None
        med = np.median(selected, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    # Split into "has a colour" vs "greyscale" pixels.
    chromatic = (sat >= 0.30) & (val >= 0.20)
    chromatic_share = float(chromatic.mean())

    if chromatic_share < 0.28:
        # Mostly grey/white/black. Name it by brightness instead.
        mean_val = float(val.mean())
        swatch = median_rgb(~chromatic if (~chromatic).any() else slice(None))
        if mean_val < 0.22:
            return "black", swatch
        if mean_val < 0.45:
            return "dark grey", swatch
        if mean_val < 0.72:
            return "grey", swatch
        return "white", swatch

    # Histogram the hue of the coloured pixels and take the strongest band.
    band_names = np.empty(hue.shape[0], dtype=object)
    band_names[:] = None
    for name, low, high in _HUE_BANDS:
        band_names[chromatic & (hue >= low) & (hue < high)] = name

    coloured = band_names[chromatic]
    if coloured.size == 0:
        return None, None
    values, counts = np.unique(coloured.astype(str), return_counts=True)
    best_index = int(counts.argmax())
    best, best_count = str(values[best_index]), int(counts[best_index])

    # If no single hue owns a decent share, the patch is too busy to name.
    if best_count / coloured.size < 0.42:
        return None, None

    winning = chromatic & (band_names == best)
    swatch = median_rgb(winning)
    mean_val = float(val[winning].mean())
    mean_sat = float(sat[winning].mean())

    # A dark or washed-out orange is a tan/brown in ordinary speech, not an
    # orange -- think a beige coat rather than a traffic cone.
    if best == "orange" and (mean_val < 0.55 or mean_sat < 0.45):
        return "brown", swatch
    if best == "yellow" and mean_val < 0.45:
        return "olive", swatch
    if mean_val < 0.32:
        return f"dark {best}", swatch
    if mean_val > 0.88 and mean_sat < 0.5:
        return f"pale {best}", swatch
    return best, swatch


def dominant_color(patch: np.ndarray, min_pixels: int = 40) -> Optional[str]:
    """Just the colour name for a patch, or None if it is not clear-cut."""
    return analyse(patch, min_pixels)[0]


def hair_tone(patch: np.ndarray) -> Optional[str]:
    """Coarse light/dark description for the top of a person's box."""
    if patch is None or patch.size == 0:
        return None
    flat = patch.reshape(-1, 3)
    if flat.shape[0] < 30:
        return None
    hsv = _rgb_to_hsv(flat)
    val = float(np.median(hsv[:, 2]))
    sat = float(np.median(hsv[:, 1]))
    hue = float(np.median(hsv[:, 0]))

    if val < 0.28:
        return "dark"
    if val > 0.68 and sat < 0.35:
        return "light"
    if sat > 0.35 and (hue < 30 or hue > 340) and val < 0.6:
        return "red"
    if sat > 0.25 and 20 <= hue <= 60 and val > 0.5:
        return "fair"
    return None
