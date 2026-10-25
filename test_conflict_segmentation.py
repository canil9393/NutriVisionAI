
import cv2

from hybrid_pipeline import run_hybrid
from food_instance_selector import (
    select_food_instances,
    resolve_known_label_conflicts,
)
from food_segmenter import segment_food

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

print("Running hybrid recognition...")
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

resolved, removed = resolve_known_label_conflicts(selected)

print("\nSegmenting original selected detections...")
before = segment_food(
    IMAGE_PATH,
    selected,
    output_dir="segmentation_results/conflict_before",
)

print("\nSegmenting resolved detections...")
after = segment_food(
    IMAGE_PATH,
    resolved,
    output_dir="segmentation_results/conflict_after",
)

print("\n--- COMPARISON ---")
print("Before:", len(before["items"]), "masks")
print("After:", len(after["items"]), "masks")
print("Before overlay:", before["overlay_path"])
print("After overlay:", after["overlay_path"])

print("\nRemoved candidates:", removed)
print("\nOpen both overlay images and compare the rice bowl.")
