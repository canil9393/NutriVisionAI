
import cv2

from hybrid_pipeline import run_hybrid
from food_instance_selector import (
    select_food_instances,
    resolve_known_label_conflicts,
)

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

print("Step 1: Running hybrid recognition...")
hybrid = run_hybrid(IMAGE_PATH)

detections = []
for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if not prediction.get("bbox"):
            continue

        item = dict(prediction)
        item["source"] = source
        detections.append(item)

print("Raw detections:", len(detections))

print("\nStep 2: Selecting detections...")
selected = select_food_instances(
    detections,
    min_confidence=0.50,
    duplicate_iou=0.85,
)
print("Selected before conflict resolution:", len(selected))

print("\nStep 3: Reviewing known rice/chutney conflicts...")
resolved, removed = resolve_known_label_conflicts(selected)

print("Selected after conflict resolution:", len(resolved))

print("\nRetained detections:")
for item in resolved:
    print(
        f"{item.get('food')} | "
        f"source={item.get('source')} | "
        f"confidence={float(item.get('confidence', 0)):.3f} | "
        f"bbox={item.get('bbox')}"
    )

print("\nRemoved candidates:")
for item in removed:
    print(item)

print("\nDiagnostic test completed.")
print("No segmentation was run and no main pipeline files were changed.")
