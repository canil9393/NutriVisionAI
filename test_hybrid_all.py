from hybrid_detector import hybrid_decision


# ============================================================
# Module 3C - Hybrid Testing
# ============================================================

print("=" * 70)
print("       NUTRIVISION AI - HYBRID DECISION ENGINE TEST")
print("=" * 70)


# ------------------------------------------------------------
# Test cases based on our actual model results
# ------------------------------------------------------------

test_cases = {

    "chapathi.png": {
        "yolo": [
            {"food": "chapati", "confidence": 0.8581},
            {"food": "chapati", "confidence": 0.4126}
        ],
        "api1": [
            {"food": "Kheer", "confidence": 0.8900}
        ],
        "api2": [
            {"food": "CHAPATHI", "confidence": 0.8548}
        ]
    },

    "dosa.png": {
        "yolo": [],
        "api1": [
            {"food": "CoconutChutney", "confidence": 0.9926}
        ],
        "api2": [
            {"food": "DOSA", "confidence": 0.8038}
        ]
    },

    "dosaj.jpg": {
        "yolo": [
            {"food": "omelette", "confidence": 0.8004}
        ],
        "api1": [
            {"food": "CoconutChutney", "confidence": 0.9952}
        ],
        "api2": [
            {"food": "DOSA", "confidence": 0.9056}
        ]
    },

    "food.png": {
        "yolo": [
            {"food": "biryani", "confidence": 0.9327}
        ],
        "api1": [
            {"food": "Biryani", "confidence": 0.9922}
        ],
        "api2": [
            {"food": "EGG", "confidence": 0.9169}
        ]
    },

    "idle.jpg": {
        "yolo": [
            {"food": "rasgulla", "confidence": 0.7485},
            {"food": "rasgulla", "confidence": 0.5042},
            {"food": "rasgulla", "confidence": 0.4498},
            {"food": "plainrice", "confidence": 0.2974}
        ],
        "api1": [
            {"food": "Idli", "confidence": 0.9978},
            {"food": "Idli", "confidence": 0.9972},
            {"food": "Idli", "confidence": 0.9960}
        ],
        "api2": [
            {"food": "IDLI", "confidence": 0.8174}
        ]
    },

    "poha.png": {
        "yolo": [
            {"food": "poha", "confidence": 0.7328}
        ],
        "api1": [
            {"food": "Poha", "confidence": 0.9987}
        ],
        "api2": [
            {"food": "EGG", "confidence": 0.8897}
        ]
    },

    "poori.jpg": {
        "yolo": [],
        "api1": [
            {"food": "Bhatura", "confidence": 0.9944}
        ],
        "api2": [
            {"food": "POORI", "confidence": 0.9167}
        ]
    },

    "samosa.jpg": {
        "yolo": [
            {"food": "alooparatha", "confidence": 0.7348}
        ],
        "api1": [
            {"food": "Samosa", "confidence": 0.9897},
            {"food": "Samosa", "confidence": 0.9888},
            {"food": "Samosa", "confidence": 0.9532}
        ],
        "api2": [
            {"food": "GreenChutney", "confidence": 0.8344}
        ]
    },

    "voda.jpg": {
        "yolo": [
            {"food": "chickentikka", "confidence": 0.5628}
        ],
        "api1": [],
        "api2": [
            {"food": "VADA", "confidence": 0.9082}
        ]
    }
}


# ------------------------------------------------------------
# Run hybrid decision
# ------------------------------------------------------------

for image_name, data in test_cases.items():

    print("\n" + "=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    result = hybrid_decision(
        data["yolo"],
        data["api1"],
        data["api2"]
    )

    print("\nYOLO11:")
    if data["yolo"]:
        for p in data["yolo"]:
            print(
                f"  {p['food']} - "
                f"{p['confidence'] * 100:.2f}%"
            )
    else:
        print("  No detection")

    print("\nAPI 1:")
    if data["api1"]:
        for p in data["api1"]:
            print(
                f"  {p['food']} - "
                f"{p['confidence'] * 100:.2f}%"
            )
    else:
        print("  No detection")

    print("\nAPI 2:")
    if data["api2"]:
        for p in data["api2"]:
            print(
                f"  {p['food']} - "
                f"{p['confidence'] * 100:.2f}%"
            )
    else:
        print("  No detection")

    print("\n" + "-" * 70)
    print("HYBRID FINAL RESULT")
    print("-" * 70)

    print("Food:", result["food"])
    print("Confidence:", round(result["confidence"], 2), "%")

    print("\nScores:")
    for food, score in result["scores"].items():
        print(f"  {food}: {score:.4f}")


print("\n" + "=" * 70)
print("              HYBRID TEST COMPLETED")
print("=" * 70)