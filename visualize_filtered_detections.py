
import cv2

from hybrid_pipeline import run_hybrid
from detection_filter import filter_detections

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"
OUTPUT_PATH = r"D:\NutriVisionAI\segmentation_results\meal_filtered_boxes.jpg"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]
hybrid = run_hybrid(IMAGE_PATH)

detections = []

for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if prediction.get("bbox"):
            item = dict(prediction)
            item["source"] = source
            detections.append(item)

filtered = filter_detections(
    detections,
    min_confidence=0.50,
    duplicate_iou=0.90,
    image_width=width,
    image_height=height,
)

colors = {
    "yolo": (0, 255, 255),
    "api1": (255, 100, 0),
    "api2": (255, 0, 255),
}

for item in filtered:
    x1, y1, x2, y2 = map(int, item["bbox"])
    color = colors.get(item["source"], (255, 255, 255))

    label = (
        f"{item['source']}:{item['food']} "
        f"{float(item['confidence']):.2f}"
    )

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(OUTPUT_PATH, image)

print("Filtered detections:", len(filtered))
print("Saved image:", OUTPUT_PATH)
