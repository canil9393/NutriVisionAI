
from food_instance_selector import (
    resolve_known_label_conflicts,
    box_iou,
)

# Simulate the rice/chutney conflict found in your real output.
test_detections = [
    {
        "food": "RICE",
        "source": "api2",
        "confidence": 0.928,
        "bbox": [232, 124, 398, 278],
    },
    {
        "food": "CoconutChutney",
        "source": "api1",
        "confidence": 0.664,
        "bbox": [232, 121, 399, 279],
    },
    {
        "food": "chapati",
        "source": "yolo",
        "confidence": 0.739,
        "bbox": [119, 174, 254, 320],
    },
]

print("Before:", len(test_detections))

selected, removed = resolve_known_label_conflicts(
    test_detections
)

print("\nAfter:", len(selected))
print("Retained detections:")
for item in selected:
    print(
        item["food"],
        "| source:", item["source"],
        "| confidence:", item["confidence"],
    )

print("\nRemoved candidates:")
for item in removed:
    print(item)

# Verify that the original input was not modified.
assert len(test_detections) == 3
assert any(item["food"] == "RICE" for item in selected)
assert not any(
    item["food"] == "CoconutChutney" for item in selected
)
print("\nTest passed.")
