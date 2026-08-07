from ultralytics import YOLO


class FoodDetector:

    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        return results

    