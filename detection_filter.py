
from copy import deepcopy
import re


# Food labels that mean the same thing.
ALIASES = {
    "plainrice": "rice",
    "whiterice": "rice",
    "rice": "rice",
    "chapathi": "chapati",
    "chappathi": "chapati",
    "chapati": "chapati",
    "puri": "poori",
    "poori": "poori",
    "idly": "idli",
    "idli": "idli",
}


def normalize_food_name(name):
    name = re.sub(r"[\s_-]+", "", str(name).lower().strip())
    return ALIASES.get(name, name)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def filter_detections(
    detections,
    min_confidence=0.50,
    duplicate_iou=0.90,
    image_width=None,
    image_height=None,
):
    """
    Filter invalid/low-confidence detections and suppress near-identical
    boxes only when their normalized labels match.

    This is a conservative baseline, not a guarantee of correct food identity.
    """

    valid = []

    for original in detections:
        detection = deepcopy(original)

        try:
            confidence = float(detection.get("confidence", 0))
            bbox = detection.get("bbox")

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(float, bbox)

        except (TypeError, ValueError):
            continue

        if confidence < min_confidence:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        if image_width is not None and image_height is not None:
            if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
                continue

            x1 = max(0, min(x1, image_width))
            x2 = max(0, min(x2, image_width))
            y1 = max(0, min(y1, image_height))
            y2 = max(0, min(y2, image_height))

            if x2 <= x1 or y2 <= y1:
                continue

        detection["bbox"] = [x1, y1, x2, y2]
        detection["normalized_food"] = normalize_food_name(
            detection.get("food", "unknown")
        )

        valid.append(detection)

    # Higher-confidence detections are considered first.
    valid.sort(
        key=lambda d: float(d["confidence"]),
        reverse=True,
    )

    kept = []

    for detection in valid:
        duplicate = False

        for existing in kept:
            same_food = (
                detection["normalized_food"]
                == existing["normalized_food"]
            )

            same_box = (
                box_iou(detection["bbox"], existing["bbox"])
                >= duplicate_iou
            )

            if same_food and same_box:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept
