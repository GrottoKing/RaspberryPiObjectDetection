"""COCO class names and the category taxonomy.

Categories are not hardcoded into the UI -- the web page only ever shows the
categories that have actually turned up in the log, so the sidebar grows as the
camera meets more of the world.
"""

from __future__ import annotations

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# label -> category. Anything unmapped falls through to "Other".
_CATEGORY_MAP = {
    "People": ["person"],
    "Vehicles": [
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat",
    ],
    "Animals": [
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "teddy bear",
    ],
    "Food & Drink": [
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "bottle", "wine glass", "cup",
    ],
    "Kitchen & Tableware": [
        "fork", "knife", "spoon", "bowl", "microwave", "oven", "toaster",
        "sink", "refrigerator",
    ],
    "Furniture": [
        "chair", "couch", "bed", "dining table", "toilet", "bench",
        "potted plant",
    ],
    "Electronics": [
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    ],
    "Personal Items": [
        "backpack", "umbrella", "handbag", "tie", "suitcase", "book", "clock",
        "vase", "scissors", "hair drier", "toothbrush",
    ],
    "Sports & Outdoors": [
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
    ],
    "Street & Signs": [
        "traffic light", "fire hydrant", "stop sign", "parking meter",
    ],
}

LABEL_TO_CATEGORY = {
    label: category
    for category, labels in _CATEGORY_MAP.items()
    for label in labels
}

# Categories are ordered by how interesting they usually are, not
# alphabetically. Unknown categories sort to the end.
CATEGORY_ORDER = [
    "People", "Animals", "Vehicles", "Electronics", "Food & Drink",
    "Kitchen & Tableware", "Furniture", "Personal Items",
    "Sports & Outdoors", "Street & Signs", "Other",
]


def category_for(label: str) -> str:
    """Map a detector label to a display category."""
    return LABEL_TO_CATEGORY.get(label.lower().strip(), "Other")


def category_sort_key(category: str) -> tuple:
    try:
        return (0, CATEGORY_ORDER.index(category), category)
    except ValueError:
        return (1, 0, category)
