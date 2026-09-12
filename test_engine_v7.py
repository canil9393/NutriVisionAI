# ============================================================
# NUTRIVISION AI - HYBRID ENGINE V7 TEST
# ============================================================

from hybrid_engine_v7 import hybrid_decision


TEST_DATA = {

    "chapathi.png": {
        "yolo": [
            {"food": "chapati", "confidence": 0.8581},
            {"food": "chapati", "confidence": 0.4126},
        ],
        "api1": [
            {"food": "Kheer", "confidence": 0.8900},
        ],
        "api2": [
            {"food": "CHAPATHI", "confidence": 0.8548},
        ],
    },

    "dosa.png": {
        "yolo": [],
        "api1": [
            {"food": "CoconutChutney", "confidence": 0.9926},
        ],
        "api2": [
            {"food": "SAMBAR", "confidence": 0.8772},
            {"food": "WHITE CHUTNEY", "confidence": 0.8763},
            {"food": "RED CHUTNEY", "confidence": 0.8086},
            {"food": "DOSA", "confidence": 0.8038},
            {"food": "CURRY", "confidence": 0.7691},
        ],
    },

    "dosaj.jpg": {
        "yolo": [
            {"food": "omelette", "confidence": 0.8004},
        ],
        "api1": [
            {"food": "CoconutChutney", "confidence": 0.9952},
        ],
        "api2": [
            {"food": "DOSA", "confidence": 0.9056},
        ],
    },

    "food.png": {
        "yolo": [
            {"food": "biryani", "confidence": 0.9327},
        ],
        "api1": [
            {"food": "Biryani", "confidence": 0.9922},
        ],
        "api2": [
            {"food": "EGG", "confidence": 0.9169},
        ],
    },

    "idle.jpg": {
        "yolo": [
            {"food": "rasgulla", "confidence": 0.7485},
            {"food": "rasgulla", "confidence": 0.5042},
            {"food": "rasgulla", "confidence": 0.4498},
            {"food": "plainrice", "confidence": 0.2974},
        ],
        "api1": [
            {"food": "Idli", "confidence": 0.9978},
            {"food": "Idli", "confidence": 0.9972},
            {"food": "Idli", "confidence": 0.9960},
        ],
        "api2": [
            {"food": "IDLI", "confidence": 0.8174},
        ],
    },

    "meal.png": {
        "yolo": [
            {"food": "plainrice", "confidence": 0.8653},
            {"food": "chapati", "confidence": 0.7395},
            {"food": "chapati", "confidence": 0.6028},
        ],
        "api1": [
            {"food": "Bhatura", "confidence": 0.9938},
            {"food": "Chole", "confidence": 0.9888},
            {"food": "Bhatura", "confidence": 0.9788},
            {"food": "WhiteRice", "confidence": 0.8861},
            {"food": "CoconutChutney", "confidence": 0.6640},
        ],
        "api2": [
            {"food": "PICKLE", "confidence": 0.9531},
            {"food": "RICE", "confidence": 0.9278},
            {"food": "PICKLE", "confidence": 0.9064},
            {"food": "PAPADAM", "confidence": 0.6388},
            {"food": "CURD", "confidence": 0.5592},
            {"food": "SAMBAR", "confidence": 0.5467},
        ],
    },

    "poha.png": {
        "yolo": [
            {"food": "poha", "confidence": 0.7328},
        ],
        "api1": [
            {"food": "Poha", "confidence": 0.9987},
        ],
        "api2": [
            {"food": "EGG", "confidence": 0.8897},
        ],
    },

    "poori.jpg": {
        "yolo": [],
        "api1": [
            {"food": "Bhatura", "confidence": 0.9944},
            {"food": "Bhatura", "confidence": 0.9918},
            {"food": "CoconutChutney", "confidence": 0.9805},
            {"food": "Bhatura", "confidence": 0.9781},
        ],
        "api2": [
            {"food": "POORI", "confidence": 0.9167},
            {"food": "SAMBAR", "confidence": 0.8636},
        ],
    },

    "samosa.jpg": {
        "yolo": [
            {"food": "alooparatha", "confidence": 0.7348},
        ],
        "api1": [
            {"food": "GreenChutney", "confidence": 0.9941},
            {"food": "Samosa", "confidence": 0.9897},
            {"food": "Samosa", "confidence": 0.9888},
            {"food": "Samosa", "confidence": 0.9532},
        ],
        "api2": [
            {"food": "GREEN CHUTNEY", "confidence": 0.8344},
        ],
    },

    "voda.jpg": {
        "yolo": [
            {"food": "chickentikka", "confidence": 0.5628},
        ],
        "api1": [],
        "api2": [
            {"food": "VADA", "confidence": 0.9082},
            {"food": "WHITE CHUTNEY", "confidence": 0.5098},
        ],
    },
}


print("=" * 70)
print("       NUTRIVISION AI - HYBRID ENGINE V7 TEST")
print("=" * 70)


for image, data in TEST_DATA.items():

    print("\n" + "=" * 70)
    print("IMAGE:", image)
    print("=" * 70)

    result = hybrid_decision(
        data["yolo"],
        data["api1"],
        data["api2"]
    )

    print("\nV7 FINAL RESULT")
    print("-" * 70)

    print("Food:", result["food"])
    print(
        "Confidence:",
        result["confidence"],
        "%"
    )

    print(
        "Supporting Models:",
        result["supporting_models"]
    )

    print(
        "Winner Margin:",
        result["winner_margin"]
    )

    print("\nScores:")

    for food, score in result["scores"].items():

        print(
            f"  {food}: {score:.4f}"
        )


print("\n" + "=" * 70)
print("              V7 TEST COMPLETED")
print("=" * 70)