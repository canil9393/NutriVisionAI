
import cv2

from hybrid_pipeline import run_hybrid
from food_instance_selector import select_food_instances

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]
hybrid = run_hybrid(IMAGE_PATH)

detections = []

for source in ("yolo", "api1", "api2"):
    for prediction in hybrid.get(source, []):
        if not prediction.get("bbox"):
            continue

        item = dict(prediction)
        item["source"] = source
        detections.append(item)

selected = select_food_instances(detections)

print("\nRaw detections:", len(detections))
print("Selected detections:", len(selected))

colors = {
    "yolo": (0, 255, 255),
    "api1": (255, 100, 0),
    "api2": (255, 0, 255),
}

for item in selected:
    x1, y1, x2, y2 = map(int, item["bbox"])
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

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

output_path = (
    r"D:\NutriVisionAI\segmentation_results"
    r"\meal_selected_instances.jpg"
)
cv2.imwrite(output_path, image)

for item in selected:
    print(
        item["source"],
        item["food"],
        item["normalized_food"],
        round(float(item["confidence"]), 3),
        [round(v, 1) for v in item["bbox"]],
    )

print("\nSaved visualization:", output_path)
