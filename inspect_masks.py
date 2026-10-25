
from pathlib import Path
import cv2
import numpy as np

MASK_DIR = Path("segmentation_results")

print("Inspecting individual mask files...\n")

for path in sorted(MASK_DIR.glob("*_mask.png")):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Could not read: {path.name}")
        continue

    values = np.unique(mask)
    nonzero = np.count_nonzero(mask)
    total = mask.size

    print(
        f"{path.name:30} "
        f"size={mask.shape[1]}x{mask.shape[0]} | "
        f"nonzero={nonzero} | "
        f"fraction={nonzero / total:.4f} | "
        f"values={values.tolist()}"
    )
