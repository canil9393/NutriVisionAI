
import cv2
from food_segmenter import get_adaptive_prompt_points

IMAGE_PATH = r"D:\NutriVisionAI\test_images\meal.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]

test_boxes = {
    "left_flatbread": [25, 202, 190, 406],
    "right_flatbread": [115, 165, 269, 328],
    "rice": [232, 124, 398, 278],
}

for name, bbox in test_boxes.items():
    point = get_adaptive_prompt_points(bbox, width, height)
    x, y = point[0]
    print(f"{name}: point=({x}, {y})")
    assert 0 <= x < width
    assert 0 <= y < height

print("Adaptive prompt coordinate tests passed.")
