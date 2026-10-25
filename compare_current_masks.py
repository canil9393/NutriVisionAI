
from pathlib import Path
import cv2
import numpy as np

folder = Path("segmentation_results/current_run")
files = sorted(folder.glob("*_mask.png"))
masks = {}

for path in files:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        masks[path.name] = image > 0

print(f"Masks found in current run: {len(masks)}")
print("\n--- HIGH OVERLAP IN CURRENT RUN ---")

names = list(masks)
found = False

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = masks[names[i]], masks[names[j]]
        intersection = np.count_nonzero(a & b)
        union = np.count_nonzero(a | b)
        iou = intersection / union if union else 0.0

        if iou >= 0.90:
            print(f"{names[i]} vs {names[j]} | IoU={iou:.4f}")
            found = True

if not found:
    print("No mask pairs with IoU >= 0.90.")
