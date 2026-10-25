
import numpy as np


def mask_iou(mask_a, mask_b):
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)

    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)

    return intersection / union if union else 0.0


def normalize_food(name):
    name = str(name).lower().strip()

    aliases = {
        "plainrice": "rice",
        "whiterice": "rice",
        "rice": "rice",
        "chapathi": "chapati",
        "chappathi": "chapati",
        "bhatura": "bhatura",
        "papadam": "papadam",
        "papad": "papadam",
        "coconut chutney": "coconutchutney",
        "coconutchutney": "coconutchutney",
    }

    return aliases.get(name, name)


def review_mask_overlaps(items, iou_threshold=0.90):
    """
    Classify overlapping masks for review.
    Does not delete masks or alter the original list.
    """
    findings = []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            iou = mask_iou(items[i]["mask"], items[j]["mask"])

            if iou < iou_threshold:
                continue

            food_a = normalize_food(items[i]["food"])
            food_b = normalize_food(items[j]["food"])

            same_label = food_a == food_b

            findings.append({
                "first_index": i,
                "second_index": j,
                "first_food": items[i]["food"],
                "second_food": items[j]["food"],
                "iou": round(iou, 4),
                "same_normalized_label": same_label,
                "status": (
                    "POSSIBLE_DUPLICATE"
                    if same_label
                    else "LABEL_CONFLICT_REVIEW"
                ),
            })

    return findings
