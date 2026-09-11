import os
import base64
import requests

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY is not set.")

IMAGE_FOLDER = r"D:\NutriVisionAI\test_images"

# API 2 - South Indian Food Classification
ROBOFLOW_URL = (
    "https://serverless.roboflow.com/"
    "my-first-project-4cb9f/1"
)

# --------------------------------------------------
# FUNCTION: TEST ROBOFLOW API 2
# --------------------------------------------------

def test_roboflow(image_path):

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

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
        return {
            "error": response.status_code,
            "message": response.text
        }

    return response.json()


# --------------------------------------------------
# TEST IMAGES
# --------------------------------------------------

image_files = [
    "dosa.png",
    "dosaj.jpg",
    "idle.jpg",
    "poori.jpg",
    "samosa.jpg",
    "voda.jpg",
    "chapathi.png",
    "food.png",
    "poha.png",
]


# --------------------------------------------------
# RUN TEST
# --------------------------------------------------

print("=" * 70)
print("       NUTRIVISION AI - ROBOFLOW API 2 TEST")
print("=" * 70)

for image_name in image_files:

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        continue

    result = test_roboflow(image_path)

    if "error" in result:

        print("API ERROR:", result["error"])
        print(result["message"])
        continue

    predictions = result.get("predictions", [])

    if not predictions:

        print("No detection")

    else:

        for prediction in predictions:

            food = prediction.get("class", "Unknown")
            confidence = prediction.get("confidence", 0)

            print(
                f"  {food} - "
                f"{confidence * 100:.2f}%"
            )


print("\n" + "=" * 70)
print("API 2 TEST COMPLETED")
print("=" * 70)