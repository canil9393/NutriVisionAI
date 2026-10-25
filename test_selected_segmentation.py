
import cv2

from hybrid_pipeline import run_hybrid
from food_instance_selector import select_food_instances
from food_segmenter import segment_food

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]

print("Step 1: Run hybrid recognition")
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

print("\nStep 2: Select food instances")
selected = select_food_instances(
    detections,
    min_confidence=0.50,
    duplicate_iou=0.85,
)

print("Selected detections:", len(selected))

print("\nStep 3: Segment selected detections")
result = segment_food(
    IMAGE_PATH,
    selected,
    output_dir="segmentation_results/current_run",
)

print("\nSegmentation result:", result["message"])
print("Masks generated:", len(result["items"]))

for item in result["items"]:
    print(
        f"{item['food']} | "
        f"source={item.get('source', 'unknown')} | "
        f"area={item['mask_area_pixels']} pixels"
    )

print("Overlay:", result["overlay_path"])
