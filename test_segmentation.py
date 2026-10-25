
from hybrid_pipeline import run_yolo
from food_segmenter import segment_food

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

print("Step 1: Detecting food with YOLO11...")
detections = run_yolo(IMAGE_PATH)

for item in detections:
    print(
        f"Food: {item['food']} | "
        f"Confidence: {item['confidence']:.2%} | "
        f"Box: {item['bbox']}"
    )

print("\nStep 2: Running segmentation...")
result = segment_food(IMAGE_PATH, detections)

print("\nResult:", result["message"])

for item in result["items"]:
    print(
        f"Food: {item['food']} | "
        f"Mask area: {item['mask_area_pixels']} pixels"
    )

if result["overlay_path"]:
    print("\nSegmented image saved at:")
    print(result["overlay_path"])
