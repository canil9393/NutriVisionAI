
import cv2

from hybrid_pipeline import run_hybrid
from food_instance_selector import (
    select_food_instances,
    box_iou,
    review_cross_label_overlaps,
)

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

hybrid = run_hybrid(IMAGE_PATH)

detections = []

for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if not prediction.get("bbox"):
            continue

        item = dict(prediction)
        item["source"] = source
        detections.append(item)

selected = select_food_instances(
    detections,
    min_confidence=0.50,
    duplicate_iou=0.85,
)

print(f"\nSelected detections: {len(selected)}")
print("\n--- CROSS-LABEL CONFLICT REVIEW ---")

conflicts = review_cross_label_overlaps(selected, iou_threshold=0.85)

if not conflicts:
    print("No high-overlap cross-label conflicts found.")
else:
    for conflict in conflicts:
        print(
            f"{conflict['food_a']} ({conflict['source_a']}) vs "
            f"{conflict['food_b']} ({conflict['source_b']}) | "
            f"IoU={conflict['iou']} | {conflict['status']}"
        )
print("\n--- OVERLAPPING DETECTION BOXES ---")

found = False

for i in range(len(selected)):
    for j in range(i + 1, len(selected)):
        a = selected[i]
        b = selected[j]

        overlap = box_iou(a["bbox"], b["bbox"])

        if overlap >= 0.50:
            print(
    f"{a.get('food', 'unknown')} ({a.get('source', 'unknown')}, "
    f"conf={float(a.get('confidence', 0)):.3f}) vs "
    f"{b.get('food', 'unknown')} ({b.get('source', 'unknown')}, "
    f"conf={float(b.get('confidence', 0)):.3f}) | "
    f"box IoU={overlap:.4f} | "
    f"A={a['bbox']} | B={b['bbox']}"
)
            found = True

if not found:
    print("No box pairs with IoU >= 0.50.")
