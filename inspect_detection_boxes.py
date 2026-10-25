
from test_selected_segmentation import IMAGE_PATH
from hybrid_pipeline import run_hybrid
from food_instance_selector import select_food_instances

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

print("\n--- SELECTED DETECTION BOXES ---")

for i, item in enumerate(selected, start=1):
    print(
        f"{i}. food={item.get('class_name', item.get('food', item.get('class', 'unknown')))} "
        f"| source={item.get('source')} "
        f"| confidence={item.get('confidence')} "
        f"| bbox={item.get('bbox')}"
    )
