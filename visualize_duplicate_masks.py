
from pathlib import Path
import cv2
import numpy as np

IMAGE_PATH = Path(r"D:\NutriVisionAI\test_images\meal.png")
MASK_DIR = Path("segmentation_results/conflict_before")
OUTPUT_PATH = MASK_DIR / "duplicate_mask_comparison.jpg"

original = cv2.imread(str(IMAGE_PATH))
if original is None:
    raise FileNotFoundError(IMAGE_PATH)

pairs = [
    ("1_bhatura_mask.png", "10_chapati_mask.png"),
    ("3_bhatura_mask.png", "7_chapati_mask.png"),
    ("5_rice_mask.png", "8_coconutchutney_mask.png"),
]

panels = []

for first, second in pairs:
    path_a = MASK_DIR / first
    path_b = MASK_DIR / second

    mask_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)

    if mask_a is None or mask_b is None:
        print("Missing mask:", first, "or", second)
        continue

    overlay = original.copy()
    a = mask_a > 0
    b = mask_b > 0

    overlay[a & ~b] = (255, 0, 0)
    overlay[b & ~a] = (0, 0, 255)
    overlay[a & b] = (0, 255, 0)

    cv2.putText(
        overlay, f"A: {first}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    cv2.putText(
        overlay, f"B: {second}", (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    panels.append(overlay)

if panels:
    comparison = cv2.vconcat(panels)
    cv2.imwrite(str(OUTPUT_PATH), comparison)
    print("Saved:", OUTPUT_PATH)
    print("Green = overlapping pixels")
    print("Blue = mask A only")
    print("Red = mask B only")
else:
    print("No comparison panels generated.")
