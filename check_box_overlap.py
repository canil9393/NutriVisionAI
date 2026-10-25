
from hybrid_pipeline import run_yolo

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

detections = run_yolo(IMAGE_PATH)


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


for i in range(len(detections)):
    for j in range(i + 1, len(detections)):
        a = detections[i]
        b = detections[j]

        iou = calculate_iou(a["bbox"], b["bbox"])

        print(
            f"{a['food']} vs {b['food']}: "
            f"IoU = {iou:.3f}"
        )
