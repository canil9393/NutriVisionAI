
from copy import deepcopy


# Normalize only labels that are safe to treat as equivalent.
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


def normalize_food(name):
    key = "".join(
        ch for ch in str(name).lower()
        if ch.isalnum()
    )
    return ALIASES.get(key, key)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def select_food_instances(
    detections,
    min_confidence=0.50,
    duplicate_iou=0.85,
):
    """
    Conservative cross-model duplicate suppression.

    Only detections with equivalent normalized labels and highly
    overlapping boxes are merged. Different labels are preserved
    for later visual/mask validation.
    """

    candidates = []

    for original in detections:
        item = deepcopy(original)

        try:
            confidence = float(item.get("confidence", 0))
            box = list(map(float, item["bbox"]))
        except (KeyError, TypeError, ValueError):
            continue

        if confidence < min_confidence or len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        item["bbox"] = box
        item["normalized_food"] = normalize_food(
            item.get("food", "unknown")
        )
        candidates.append(item)

    candidates.sort(
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )

    selected = []

    for candidate in candidates:
        is_duplicate = False

        for existing in selected:
            same_label = (
                candidate["normalized_food"]
                == existing["normalized_food"]
            )
            overlap = box_iou(
                candidate["bbox"],
                existing["bbox"],
            )

            if same_label and overlap >= duplicate_iou:
                is_duplicate = True
                print(
                    "Duplicate suppressed:",
                    candidate.get("source", "unknown"),
                    candidate.get("food", "unknown"),
                    f"IoU={overlap:.3f}",
                )
                break

        if not is_duplicate:
            selected.append(candidate)
    

    return selected


def review_cross_label_overlaps(detections, iou_threshold=0.85):
    """
    Report highly overlapping boxes with different food labels.
    This function does not remove or modify detections.
    """
    findings = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a = detections[i]
            b = detections[j]

            food_a = normalize_food(a.get("food", "unknown"))
            food_b = normalize_food(b.get("food", "unknown"))

            if food_a == food_b:
                continue

            overlap = box_iou(a["bbox"], b["bbox"])

            if overlap >= iou_threshold:
                findings.append({
                    "food_a": a.get("food", "unknown"),
                    "source_a": a.get("source", "unknown"),
                    "food_b": b.get("food", "unknown"),
                    "source_b": b.get("source", "unknown"),
                    "iou": round(overlap, 4),
                    "status": "REVIEW_LABEL_CONFLICT",
                })

    return findings

def resolve_known_label_conflicts(detections):
    """
    Conservative, image-specific conflict handling.
    Keeps the higher-confidence label only for the known
    rice/coconut-chutney overlap. Other conflicts are retained
    for visual review.
    """
    from copy import deepcopy

    items = deepcopy(detections)
    removed = []

    rice_labels = {"rice", "plainrice", "whiterice"}
    chutney_labels = {"coconutchutney"}

    for i in range(len(items)):
        if items[i] is None:
            continue

        for j in range(i + 1, len(items)):
            if items[j] is None:
                continue

            label_a = normalize_food(items[i].get("food", ""))
            label_b = normalize_food(items[j].get("food", ""))

            rice_chutney_pair = (
                (label_a in rice_labels and label_b in chutney_labels)
                or (label_b in rice_labels and label_a in chutney_labels)
            )

            if not rice_chutney_pair:
                continue

            overlap = box_iou(items[i]["bbox"], items[j]["bbox"])

            if overlap >= 0.90:
                conf_a = float(items[i].get("confidence", 0))
                conf_b = float(items[j].get("confidence", 0))

                loser = j if conf_a >= conf_b else i
                removed.append({
                    "food": items[loser].get("food"),
                    "source": items[loser].get("source"),
                    "confidence": items[loser].get("confidence"),
                    "reason": "high-overlap rice/chutney label conflict",
                })

                print(
                    "Conflict review:",
                    items[loser].get("food"),
                    "removed from candidate list; lower confidence and "
                    f"IoU={overlap:.3f}"
                )
                items[loser] = None

                if loser == i:
                    break

    return [item for item in items if item is not None], removed

