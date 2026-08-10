from ultralytics import YOLO
import os

model = YOLO("models/best.pt")

test_folder = "test_images"

for image in os.listdir(test_folder):

    image_path = os.path.join(test_folder, image)

    print(f"\n--- Testing: {image} ---")

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            food_name = model.names[class_id]

            print(
                f"{food_name}: "
                f"{confidence * 100:.2f}%"
            )

print("\nAll tests completed!")