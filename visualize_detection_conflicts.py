
import cv2

from hybrid_pipeline import run_hybrid
from food_instance_selector import select_food_instances

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"
OUTPUT_PATH = r"segmentation_results\current_run\detection_conflicts.jpg"

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

for index, item in enumerate(selected, start=1):
    x1, y1, x2, y2 = map(int, item["bbox"])
    label = (
        f"{index}: {item.get('food', 'unknown')} "
        f"[{item.get('source', 'unknown')}] "
        f"{float(item.get('confidence', 0)):.2f}"
    )

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        image,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(OUTPUT_PATH, image)
print(f"Saved diagnostic image: {OUTPUT_PATH}")
print(f"Boxes drawn: {len(selected)}")
