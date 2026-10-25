
from pathlib import Path
import cv2
import numpy as np

MASK_DIR = Path("segmentation_results")
masks = {}

for path in sorted(MASK_DIR.glob("*_mask.png")):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        masks[path.name] = image > 0

print("\n--- EXACT MASK COMPARISON ---")

names = list(masks.keys())
found = False

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = masks[names[i]]
        b = masks[names[j]]

        if a.shape != b.shape:
            continue

        intersection = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)
        iou = intersection / union if union else 0.0

        if np.array_equal(a, b):
            print(f"EXACT DUPLICATE: {names[i]} == {names[j]}")
            found = True
        elif iou >= 0.90:
            print(
                f"HIGH OVERLAP: {names[i]} vs {names[j]} "
                f"| IoU={iou:.4f}"
            )
            found = True

if not found:
    print("No exact duplicates or high-overlap pairs found.")
