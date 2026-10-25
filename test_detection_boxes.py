
import cv2

from hybrid_pipeline import run_yolo

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"
OUTPUT_PATH = r"D:\NutriVisionAI\segmentation_results\meal_boxes.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

detections = run_yolo(IMAGE_PATH)

for detection in detections:
    x1, y1, x2, y2 = map(int, detection["bbox"])
    food = detection["food"]
    confidence = detection["confidence"]

    # Draw the detection box.
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2,
    )

    label = f"{food}: {confidence:.2%}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

cv2.imwrite(OUTPUT_PATH, image)

print("Image dimensions:", image.shape[1], "x", image.shape[0])
print("Detection-box image saved:", OUTPUT_PATH)

for detection in detections:
    print(
        detection["food"],
        "confidence =", round(detection["confidence"], 3),
        "box =", [round(v, 1) for v in detection["bbox"]],
    )