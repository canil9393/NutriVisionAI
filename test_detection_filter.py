
import cv2

from hybrid_pipeline import run_hybrid
from detection_filter import filter_detections

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]

hybrid = run_hybrid(IMAGE_PATH)

all_detections = []

for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if not prediction.get("bbox"):
            continue

        detection = dict(prediction)
        detection["source"] = source
        all_detections.append(detection)

filtered = filter_detections(
    all_detections,
    min_confidence=0.50,
    duplicate_iou=0.90,
    image_width=width,
    image_height=height,
)

print("\nRaw detections:", len(all_detections))
print("Filtered detections:", len(filtered))

for detection in filtered:
    print(
        f"{detection['source']:5} | "
        f"{detection['food']:16} | "
        f"normalized={detection['normalized_food']:12} | "
        f"confidence={detection['confidence']:.3f} | "
        f"box={[round(v, 1) for v in detection['bbox']]}"
    )
