import os

from hybrid_pipeline import run_hybrid


IMAGE_FOLDER = r"D:\NutriVisionAI\test_images"


print("=" * 70)
print("       NUTRIVISION AI - REAL HYBRID PIPELINE TEST")
print("=" * 70)


image_files = [
    file
    for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


for image_name in image_files:

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    result = run_hybrid(image_path)

    print("\nYOLO11:")

    if result["yolo"]:

        for prediction in result["yolo"]:

            print(
                f"  {prediction['food']} - "
                f"{prediction['confidence'] * 100:.2f}%"
            )

    else:
        print("  No detection")


    print("\nAPI 1:")

    if result["api1"]:

        for prediction in result["api1"]:

            print(
                f"  {prediction['food']} - "
                f"{prediction['confidence'] * 100:.2f}%"
            )

    else:
        print("  No detection")


    print("\nAPI 2:")

    if result["api2"]:

        for prediction in result["api2"]:

            print(
                f"  {prediction['food']} - "
                f"{prediction['confidence'] * 100:.2f}%"
            )

    else:
        print("  No detection")


    final = result["final"]

    print("\n" + "-" * 70)
    print("HYBRID FINAL RESULT")
    print("-" * 70)

    print(
        "Food:",
        final["food"]
    )

    print(
        "Confidence:",
        final["confidence"],
        "%"
    )

    print("\nScores:")

    for food, score in final["scores"].items():

        print(
            f"  {food}: {score:.4f}"
        )


print("\n" + "=" * 70)
print("REAL HYBRID TEST COMPLETED")
print("=" * 70)