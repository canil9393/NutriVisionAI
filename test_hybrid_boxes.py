
import cv2

from hybrid_pipeline import run_hybrid

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"
OUTPUT_PATH = r"D:\NutriVisionAI\segmentation_results\meal_hybrid_boxes.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Run YOLO11, API1 and API2.
hybrid = run_hybrid(IMAGE_PATH)

sources = [
    ("YOLO11", hybrid.get("yolo", []), (0, 255, 255)),
    ("API1", hybrid.get("api1", []), (255, 0, 0)),
    ("API2", hybrid.get("api2", []), (0, 165, 255)),
]

height, width = image.shape[:2]

for source_name, detections, color in sources:
    print(f"\n{source_name}: {len(detections)} detections")

    for detection in detections:
        bbox = detection.get("bbox")

        if not bbox or len(bbox) != 4:
            print(
                "Skipped: missing bounding box for",
                detection.get("food", "unknown"),
            )
            continue

        x1, y1, x2, y2 = map(float, bbox)

        # Reject boxes that are outside the image or invalid.
        if x2 <= x1 or y2 <= y1:
            print("Skipped invalid box:", detection)
            continue

        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width - 1))
        y2 = max(0, min(int(y2), height - 1))

        label = (
            f"{source_name}: {detection.get('food', 'unknown')} "
            f"{float(detection.get('confidence', 0)):.2f}"
        )

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

        print(
            f"{detection.get('food', 'unknown')} | "
            f"confidence={detection.get('confidence', 0):.3f} | "
            f"bbox={[x1, y1, x2, y2]}"
        )

cv2.imwrite(OUTPUT_PATH, image)

print("\nFinal hybrid decision:", hybrid.get("final"))
print("Saved diagnostic image:", OUTPUT_PATH)
