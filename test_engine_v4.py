# ============================================================
# NUTRIVISION AI - HYBRID ENGINE V4 TEST
# ============================================================

from hybrid_engine_v4 import hybrid_decision_v4


# ------------------------------------------------------------
# Test helper
# ------------------------------------------------------------

def test_case(
    image_name,
    yolo,
    api1,
    api2
):

    print("=" * 70)
    print("IMAGE:", image_name)
    print("=" * 70)

    result = hybrid_decision_v4(
        yolo,
        api1,
        api2
    )

    print()
    print("V4 FINAL RESULT")
    print("-" * 70)

    print(
        "Food:",
        result["food"]
    )

    print(
        "Confidence:",
        result["confidence"],
        "%"
    )

    print()
    print("Scores:")

    for food, score in result["scores"].items():

        print(
            f"  {food}: {score:.4f}"
        )

    print()
    print(
        "Supporting Models:",
        result["supporting_models"]
    )

    print(
        "Winner Margin:",
        result["margin"]
    )

    print()


# ============================================================
# TEST DATA FROM YOUR REAL HYBRID TEST
# ============================================================


test_case(

    "chapathi.png",

    [
        {"food": "chapati", "confidence": 0.8581},
        {"food": "chapati", "confidence": 0.4126},
    ],

    [
        {"food": "Kheer", "confidence": 0.8900},
    ],

    [
        {"food": "CHAPATHI", "confidence": 0.8548},
    ]
)


test_case(

    "dosa.png",

    [],

    [
        {"food": "CoconutChutney", "confidence": 0.9926},
    ],

    [
        {"food": "SAMBAR", "confidence": 0.8772},
        {"food": "WHITE CHUTNEY", "confidence": 0.8763},
        {"food": "RED CHUTNEY", "confidence": 0.8086},
        {"food": "DOSA", "confidence": 0.8038},
        {"food": "CURRY", "confidence": 0.7691},
    ]
)


test_case(

    "dosaj.jpg",

    [
        {"food": "omelette", "confidence": 0.8004},
    ],

    [
        {"food": "CoconutChutney", "confidence": 0.9952},
    ],

    [
        {"food": "DOSA", "confidence": 0.9056},
    ]
)


test_case(

    "food.png",

    [
        {"food": "biryani", "confidence": 0.9327},
    ],

    [
        {"food": "Biryani", "confidence": 0.9922},
    ],

    [
        {"food": "EGG", "confidence": 0.9169},
    ]
)


test_case(

    "idle.jpg",

    [
        {"food": "rasgulla", "confidence": 0.7485},
        {"food": "rasgulla", "confidence": 0.5042},
        {"food": "rasgulla", "confidence": 0.4498},
        {"food": "plainrice", "confidence": 0.2974},
    ],

    [
        {"food": "Idli", "confidence": 0.9978},
        {"food": "Idli", "confidence": 0.9972},
        {"food": "Idli", "confidence": 0.9960},
    ],

    [
        {"food": "IDLI", "confidence": 0.8174},
    ]
)


test_case(

    "meal.png",

    [
        {"food": "plainrice", "confidence": 0.8653},
        {"food": "chapati", "confidence": 0.7395},
        {"food": "chapati", "confidence": 0.6028},
    ],

    [
        {"food": "Bhatura", "confidence": 0.9938},
        {"food": "Chole", "confidence": 0.9888},
        {"food": "Bhatura", "confidence": 0.9788},
        {"food": "WhiteRice", "confidence": 0.8861},
        {"food": "CoconutChutney", "confidence": 0.6640},
    ],

    [
        {"food": "PICKLE", "confidence": 0.9531},
        {"food": "RICE", "confidence": 0.9278},
        {"food": "PICKLE", "confidence": 0.9064},
        {"food": "PAPADAM", "confidence": 0.6388},
        {"food": "CURD", "confidence": 0.5592},
        {"food": "SAMBAR", "confidence": 0.5467},
    ]
)


test_case(

    "poha.png",

    [
        {"food": "poha", "confidence": 0.7328},
    ],

    [
        {"food": "Poha", "confidence": 0.9987},
    ],

    [
        {"food": "EGG", "confidence": 0.8897},
    ]
)


test_case(

    "poori.jpg",

    [],

    [
        {"food": "Bhatura", "confidence": 0.9944},
        {"food": "Bhatura", "confidence": 0.9918},
        {"food": "CoconutChutney", "confidence": 0.9805},
        {"food": "Bhatura", "confidence": 0.9781},
    ],

    [
        {"food": "POORI", "confidence": 0.9167},
        {"food": "SAMBAR", "confidence": 0.8636},
    ]
)


test_case(

    "samosa.jpg",

    [
        {"food": "alooparatha", "confidence": 0.7348},
    ],

    [
        {"food": "GreenChutney", "confidence": 0.9941},
        {"food": "Samosa", "confidence": 0.9897},
        {"food": "Samosa", "confidence": 0.9888},
        {"food": "Samosa", "confidence": 0.9532},
    ],

    [
        {"food": "GREEN CHUTNEY", "confidence": 0.8344},
    ]
)


test_case(

    "voda.jpg",

    [
        {"food": "chickentikka", "confidence": 0.5628},
    ],

    [],

    [
        {"food": "VADA", "confidence": 0.9082},
        {"food": "WHITE CHUTNEY", "confidence": 0.5098},
    ]
)


print("=" * 70)
print("V4 TEST COMPLETED")
print("=" * 70)