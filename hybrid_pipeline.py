import os
import base64
import requests

from ultralytics import YOLO

from hybrid_detector import hybrid_decision

from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

load_dotenv()
YOLO_MODEL = r"D:\NutriVisionAI\models\best.pt"

API_KEY = os.getenv("ROBOFLOW_API_KEY")
print("Roboflow API key loaded:", bool(API_KEY))

API1_URL = "https://serverless.roboflow.com/indianfoodnet/1"

API2_URL = "https://serverless.roboflow.com/my-first-project-4cb9f/1"


# ============================================================
# LOAD YOLO11
# ============================================================

print("Loading YOLO11 model...")

model = YOLO(YOLO_MODEL)

print("YOLO11 loaded successfully!")


# ============================================================
# YOLO11
# ============================================================


def run_yolo(image_path):
    results = model(image_path, verbose=False)
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            food_name = model.names[class_id]

            # Preserve bounding-box coordinates for segmentation
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "food": food_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })

    return detections



# ============================================================
# ROBOFLOW API
# ============================================================

def run_roboflow(image_path, api_url):

    if not API_KEY:
        return []

    try:

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        headers = {
            "Content-Type":
            "application/x-www-form-urlencoded"
        }

        params = {
            "api_key": API_KEY
        }

        response = requests.post(
            api_url,
            params=params,
            headers=headers,
            data=image_base64,
            timeout=30
        )

        
        print(f"\nRoboflow endpoint: {api_url}")
        print("HTTP status:", response.status_code)

        if response.status_code != 200:
            print("Roboflow response:", response.text[:1000])
            return []

        result = response.json()

        # Temporarily print the response structure for diagnosis.
        print("Response keys:", list(result.keys()))
        print("Raw response preview:", str(result)[:1500])

        predictions = result.get("predictions", [])
        print("Prediction count:", len(predictions))



        detections = []

        for prediction in predictions:
            food = prediction.get("class", "Unknown")
            confidence = float(prediction.get("confidence", 0))

            # Roboflow bounding-box coordinates
            x = float(prediction.get("x", 0))
            y = float(prediction.get("y", 0))
            width = float(prediction.get("width", 0))
            height = float(prediction.get("height", 0))

            if width <= 0 or height <= 0:
                continue

            detections.append({
                "food": food,
                "confidence": confidence,
                "bbox": [
                    x - width / 2,
                    y - height / 2,
                    x + width / 2,
                    y + height / 2,
                ],
            })

        return detections

    except Exception as e:
        print("API error:", e)
        return []

# ============================================================
# API 1
# ============================================================

def run_api1(image_path):

    return run_roboflow(
        image_path,
        API1_URL
    )


# ============================================================
# API 2
# ============================================================

def run_api2(image_path):

    return run_roboflow(
        image_path,
        API2_URL
    )


# ============================================================
# COMPLETE HYBRID PIPELINE
# ============================================================

def run_hybrid(image_path):

    print("\nRunning YOLO11...")
    yolo_results = run_yolo(image_path)

    print("Running API 1...")
    api1_results = run_api1(image_path)

    print("Running API 2...")
    api2_results = run_api2(image_path)

    final_result = hybrid_decision(
        yolo_results,
        api1_results,
        api2_results
    )

    return {
        "yolo": yolo_results,
        "api1": api1_results,
        "api2": api2_results,
        "final": final_result
    }