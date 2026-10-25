
from pathlib import Path

import cv2

from hybrid_pipeline import run_hybrid
from food_segmenter import segment_food

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"
OUTPUT_PATH = Path(r"D:\NutriVisionAI\segmentation_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def calculate_iou(box_a, box_b):
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

    return intersection / union if union > 0 else 0


def remove_exact_box_duplicates(detections, iou_threshold=0.90):
    """
    Suppress highly overlapping boxes.
    This is a conservative first pass, not semantic food verification.
    """

    valid = [
        dict(d)
        for d in detections
        if d.get("bbox")
        and len(d["bbox"]) == 4
        and float(d.get("confidence", 0)) >= 0.50
    ]

    valid.sort(
        key=lambda d: float(d.get("confidence", 0)),
        reverse=True,
    )

    kept = []

    for detection in valid:
        box = list(map(float, detection["bbox"]))

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        duplicate = any(
            calculate_iou(box, existing["bbox"]) >= iou_threshold
            for existing in kept
        )

        if not duplicate:
            detection["bbox"] = box
            kept.append(detection)
        else:
            print(
                "Suppressed overlapping box:",
                detection.get("source"),
                detection.get("food"),
                box,
            )

    return kept


print("1. Running hybrid recognition...")
hybrid = run_hybrid(IMAGE_PATH)

all_detections = []

for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if not prediction.get("bbox"):
            print(
                "Skipping detection without a box:",
                source,
                prediction.get("food"),
            )
            continue

        detection = dict(prediction)
        detection["source"] = source
        all_detections.append(detection)

print("\nRaw detections:", len(all_detections))

for detection in all_detections:
    print(
        detection["source"],
        detection["food"],
        round(float(detection["confidence"]), 3),
        detection["bbox"],
    )

unique_detections = remove_exact_box_duplicates(all_detections)

print("\nDetections passed to segmentation:", len(unique_detections))

for detection in unique_detections:
    print(
        detection["source"],
        detection["food"],
        detection["bbox"],
    )

print("\n2. Running SAM segmentation...")
result = segment_food(IMAGE_PATH, unique_detections)

print("\nResult:", result["message"])

for item in result["items"]:
    print(
        item["food"],
        "| source:",
        item.get("source", "not recorded"),
        "| mask area:",
        item["mask_area_pixels"],
        "pixels",
    )

print("\nCombined overlay:", result["overlay_path"])
