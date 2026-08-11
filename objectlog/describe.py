"""Turning a bare detection into a readable description.

The detector only knows 80 nouns. This module looks at the pixels inside the
box to add colour, so the log reads "Person with dark hair wearing blue"
rather than "person" eighty times in a row.

Deliberately conservative: if the pixels do not support an attribute, it is
left out rather than invented.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import colors

# Classes where a colour word in front reads naturally ("a blue car"). For
# things like "pizza" or "person" it does not, so they are handled separately.
_COLOUR_PREFIX_OK = {
    "car", "truck", "bus", "bicycle", "motorcycle", "boat", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "bottle", "cup", "bowl",
    "chair", "couch", "bed", "book", "vase", "clock", "laptop", "cell phone",
    "remote", "keyboard", "mouse", "tv", "teddy bear", "sports ball",
    "frisbee", "kite", "skateboard", "surfboard", "toothbrush", "scissors",
    "potted plant", "bench", "wine glass",
}

# Animals: "a black cat" reads fine too.
_ANIMALS = {
    "cat", "dog", "bird", "horse", "sheep", "cow", "bear", "elephant",
    "zebra", "giraffe",
}


def _crop(frame: np.ndarray, box: Tuple[float, float, float, float],
          x0f: float, y0f: float, x1f: float, y1f: float) -> Optional[np.ndarray]:
    """Crop a sub-region of a box, given fractional coordinates within it."""
    height, width = frame.shape[:2]
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    if bw <= 1 or bh <= 1:
        return None
    x0 = int(round(bx0 + bw * x0f))
    x1 = int(round(bx0 + bw * x1f))
    y0 = int(round(by0 + bh * y0f))
    y1 = int(round(by0 + bh * y1f))
    x0, x1 = max(0, min(x0, width - 1)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height - 1)), max(0, min(y1, height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return frame[y0:y1, x0:x1]


def describe(frame: np.ndarray, label: str,
             box: Tuple[float, float, float, float]) -> dict:
    """Build a description for one detection.

    Returns a dict with `text` (the headline), `attributes` (structured bits
    the UI can show as chips) and `swatch` (an #rrggbb for the colour dot).
    """
    label = label.lower().strip()
    attributes: dict[str, str] = {}
    swatch = None

    if label == "person":
        # Top-centre of the box is usually head/hair; the band below it is the
        # torso, which is the best proxy for "what are they wearing".
        head = _crop(frame, box, 0.30, 0.02, 0.70, 0.18)
        torso = _crop(frame, box, 0.22, 0.24, 0.78, 0.58)

        tone = colors.hair_tone(head) if head is not None else None
        garment, garment_rgb = (colors.analyse(torso) if torso is not None
                                else (None, None))

        if tone:
            attributes["hair"] = tone
        if garment:
            attributes["wearing"] = garment
            if garment_rgb:
                swatch = "#%02x%02x%02x" % garment_rgb

        parts = ["Person"]
        if tone:
            parts.append(f"with {tone} hair")
        if garment:
            parts.append(f"wearing {garment}")
        text = " ".join(parts) if len(parts) > 1 else "Person"
        return {"text": text, "attributes": attributes, "swatch": swatch}

    # Everything else: sample the middle of the box, away from the edges where
    # the background bleeds in.
    core = _crop(frame, box, 0.20, 0.20, 0.80, 0.80)
    colour, colour_rgb = colors.analyse(core) if core is not None else (None, None)
    if colour:
        attributes["colour"] = colour
        if colour_rgb:
            swatch = "#%02x%02x%02x" % colour_rgb

    pretty = label[0].upper() + label[1:]
    if colour and (label in _COLOUR_PREFIX_OK or label in _ANIMALS):
        text = f"{colour[0].upper()}{colour[1:]} {label}"
    elif colour:
        text = f"{pretty} ({colour})"
    else:
        text = pretty

    return {"text": text, "attributes": attributes, "swatch": swatch}


def summarise(label: str, attributes: dict) -> str:
    """Rebuild the headline from stored attributes (used by tests/tools)."""
    if label == "person":
        parts = ["Person"]
        if attributes.get("hair"):
            parts.append(f"with {attributes['hair']} hair")
        if attributes.get("wearing"):
            parts.append(f"wearing {attributes['wearing']}")
        return " ".join(parts)
    colour = attributes.get("colour")
    pretty = label[0].upper() + label[1:] if label else "Unknown"
    if colour and (label in _COLOUR_PREFIX_OK or label in _ANIMALS):
        return f"{colour[0].upper()}{colour[1:]} {label}"
    if colour:
        return f"{pretty} ({colour})"
    return pretty
