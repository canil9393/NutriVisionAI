
import cv2
import numpy as np

from hybrid_pipeline import run_hybrid
from food_instance_selector import select_food_instances
from food_segmenter import segment_food

from mask_instance_filter import review_mask_overlaps

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]
image_area = height * width

hybrid = run_hybrid(IMAGE_PATH)

detections = []
for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if prediction.get("bbox"):
            item = dict(prediction)
            item["source"] = source
            detections.append(item)

selected = select_food_instances(detections)

result = segment_food(IMAGE_PATH, selected)
items = result["items"]
duplicates = review_mask_overlaps(items)

print("\n--- DUPLICATE MASK CANDIDATES ---")
findings = review_mask_overlaps(items)

print("\n--- MASK OVERLAP REVIEW ---")

for finding in findings:
    print(
        f"{finding['first_food']} vs "
        f"{finding['second_food']} | "
        f"IoU={finding['iou']} | "
        f"{finding['status']}"
    )

print("\n--- MASK VALIDATION REPORT ---")

for i, item in enumerate(items):
    mask = item["mask"].astype(bool)
    area = int(np.count_nonzero(mask))
    image_fraction = area / image_area

    print(
        f"{i + 1}. {item['food']} "
        f"| source={item.get('source', 'unknown')} "
        f"| area={area}px "
        f"| image fraction={image_fraction:.3f}"
    )

    if image_fraction > 0.25:
        print("   REVIEW: mask covers more than 25% of image")

print("\n--- MASK OVERLAP ---")

for i in range(len(items)):
    mask_a = items[i]["mask"].astype(bool)

    for j in range(i + 1, len(items)):
        mask_b = items[j]["mask"].astype(bool)

        intersection = np.count_nonzero(mask_a & mask_b)
        union = np.count_nonzero(mask_a | mask_b)

        iou = intersection / union if union else 0.0

        if iou > 0.10:
            print(
                f"{items[i]['food']} ({i + 1}) vs "
                f"{items[j]['food']} ({j + 1}) "
                f"| mask IoU={iou:.3f}"
            )

print("\nOverlay:", result["overlay_path"])
