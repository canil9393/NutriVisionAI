from utils.food_detector import FoodDetector


detector = FoodDetector()

results = detector.detect("uploads/test.jpg")


for result in results:

    boxes = result.boxes

    for box in boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )