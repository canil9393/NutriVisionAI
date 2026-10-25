
from pathlib import Path
import cv2
import numpy as np

FOLDER = Path("segmentation_results/conflict_before")
THRESHOLD = 0.80

mask_files = sorted(FOLDER.glob("*_mask.png"))
masks = {}

for path in mask_files:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is not None:
        masks[path.stem] = image > 0

print("Loaded masks:", len(masks))
print("\nMask pairs with IoU >=", THRESHOLD)

names = list(masks)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name_a, name_b = names[i], names[j]
        mask_a, mask_b = masks[name_a], masks[name_b]

        if mask_a.shape != mask_b.shape:
            continue

        intersection = np.count_nonzero(mask_a & mask_b)
        union = np.count_nonzero(mask_a | mask_b)
        iou = intersection / union if union else 0.0

        if iou >= THRESHOLD:
            print(f"{name_a} vs {name_b}: IoU={iou:.4f}")
