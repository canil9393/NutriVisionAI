import os
import base64
import requests
from ultralytics import YOLO

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY is not set.")

YOLO_MODEL = r"D:\NutriVisionAI\models\best.pt"
IMAGE_FOLDER = r"D:\NutriVisionAI\test_images"

ROBOFLOW_URL = "https://serverless.roboflow.com/indianfoodnet/1"

# --------------------------------------------------
# LOAD YOLO11
# --------------------------------------------------

print("Loading YOLO11 model...")

model = YOLO(YOLO_MODEL)

print("YOLO11 loaded successfully!\n")


# --------------------------------------------------
# FUNCTION: YOLO11 TEST
# --------------------------------------------------

def test_yolo(image_path):

    results = model(image_path, verbose=False)

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append(
                (class_name, confidence)
            )

    return detections


# --------------------------------------------------
# FUNCTION: ROBOFLOW TEST
# --------------------------------------------------

def test_roboflow(image_path):

    # Read image
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    # Convert to Base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    params = {
        "api_key": API_KEY
    }

    response = requests.post(
        ROBOFLOW_URL,
        params=params,
        headers=headers,
        data=image_base64
    )

    if response.status_code != 200:
        return [
            ("API ERROR", response.status_code)
        ]

    result = response.json()

    predictions = result.get("predictions", [])

    detections = []

    for prediction in predictions:

        class_name = prediction.get("class", "Unknown")
        confidence = prediction.get("confidence", 0)

        detections.append(
            (class_name, confidence)
        )

    return detections


# --------------------------------------------------
# TEST ALL IMAGES
# --------------------------------------------------

image_files = [
    file for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("=" * 70)
print("       NUTRIVISION AI - HYBRID FOOD TEST")
print("=" * 70)


for image_name in image_files:

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    # -------------------------
    # YOLO11
    # -------------------------

    print("\nYOLO11:")

    yolo_results = test_yolo(image_path)

    if not yolo_results:

        print("  No detection")

    else:

        for food, confidence in yolo_results:

            print(
                f"  {food} "
                f"- {confidence * 100:.2f}%"
            )


    # -------------------------
    # ROBOFLOW
    # -------------------------

    print("\nRoboflow API:")

    roboflow_results = test_roboflow(image_path)

    if not roboflow_results:

        print("  No detection")

    else:

        for food, confidence in roboflow_results:

            if food == "API ERROR":

                print(
                    f"  API Error: {confidence}"
                )

            else:

                print(
                    f"  {food} "
                    f"- {confidence * 100:.2f}%"
                )


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)