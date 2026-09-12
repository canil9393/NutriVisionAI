# ============================================================
# NUTRIVISION AI - HYBRID ENGINE V8 TEST
# ============================================================

from hybrid_engine_v8 import hybrid_decision


# ------------------------------------------------------------
# TEST DATA
#
# These are the real model results obtained previously.
# ------------------------------------------------------------

TEST_CASES = {

    "chapathi.png": {

        "yolo": [
            {
                "food": "chapati",
                "confidence": 0.8581
            },
            {
                "food": "chapati",
                "confidence": 0.4126
            }
        ],

        "api1": [
            {
                "food": "Kheer",
                "confidence": 0.8900
            }
        ],

        "api2": [
            {
                "food": "CHAPATHI",
                "confidence": 0.8548
            }
        ]

    },


    "dosa.png": {

        "yolo": [],

        "api1": [
            {
                "food": "CoconutChutney",
                "confidence": 0.9926
            }
        ],

        "api2": [
            {
                "food": "SAMBAR",
                "confidence": 0.8772
            },
            {
                "food": "WHITE CHUTNEY",
                "confidence": 0.8763
            },
            {
                "food": "RED CHUTNEY",
                "confidence": 0.8086
            },
            {
                "food": "DOSA",
                "confidence": 0.8038
            },
            {
                "food": "CURRY",
                "confidence": 0.7691
            }
        ]

    },


    "dosaj.jpg": {

        "yolo": [
            {
                "food": "omelette",
                "confidence": 0.8004
            }
        ],

        "api1": [
            {
                "food": "CoconutChutney",
                "confidence": 0.9952
            }
        ],

        "api2": [
            {
                "food": "DOSA",
                "confidence": 0.9056
            }
        ]

    },


    "food.png": {

        "yolo": [
            {
                "food": "biryani",
                "confidence": 0.9327
            }
        ],

        "api1": [
            {
                "food": "Biryani",
                "confidence": 0.9922
            }
        ],

        "api2": [
            {
                "food": "EGG",
                "confidence": 0.9169
            }
        ]

    },


    "idle.jpg": {

        "yolo": [
            {
                "food": "rasgulla",
                "confidence": 0.7485
            },
            {
                "food": "rasgulla",
                "confidence": 0.5042
            },
            {
                "food": "rasgulla",
                "confidence": 0.4498
            },
            {
                "food": "plainrice",
                "confidence": 0.2974
            }
        ],

        "api1": [
            {
                "food": "Idli",
                "confidence": 0.9978
            },
            {
                "food": "Idli",
                "confidence": 0.9972
            },
            {
                "food": "Idli",
                "confidence": 0.9960
            },
            {
                "food": "CoconutChutney",
                "confidence": 0.9939
            },
            {
                "food": "Idli",
                "confidence": 0.9908
            }
        ],

        "api2": [
            {
                "food": "IDLI",
                "confidence": 0.8174
            },
            {
                "food": "PICKLE",
                "confidence": 0.7503
            },
            {
                "food": "GREEN CHUTNEY",
                "confidence": 0.5375
            },
            {
                "food": "GREEN CHUTNEY",
                "confidence": 0.5292
            },
            {
                "food": "WHITE CHUTNEY",
                "confidence": 0.4499
            }
        ]

    },


    "meal.png": {

        "yolo": [
            {
                "food": "plainrice",
                "confidence": 0.8653
            },
            {
                "food": "chapati",
                "confidence": 0.7395
            },
            {
                "food": "chapati",
                "confidence": 0.6028
            }
        ],

        "api1": [
            {
                "food": "Bhatura",
                "confidence": 0.9938
            },
            {
                "food": "Chole",
                "confidence": 0.9888
            },
            {
                "food": "Bhatura",
                "confidence": 0.9788
            },
            {
                "food": "WhiteRice",
                "confidence": 0.8861
            },
            {
                "food": "CoconutChutney",
                "confidence": 0.6640
            }
        ],

        "api2": [
            {
                "food": "PICKLE",
                "confidence": 0.9531
            },
            {
                "food": "RICE",
                "confidence": 0.9278
            },
            {
                "food": "PICKLE",
                "confidence": 0.9064
            },
            {
                "food": "PAPADAM",
                "confidence": 0.6388
            },
            {
                "food": "CURD",
                "confidence": 0.5592
            },
            {
                "food": "SAMBAR",
                "confidence": 0.5467
            }
        ]

    },


    "poha.png": {

        "yolo": [
            {
                "food": "poha",
                "confidence": 0.7328
            }
        ],

        "api1": [
            {
                "food": "Poha",
                "confidence": 0.9987
            }
        ],

        "api2": [
            {
                "food": "EGG",
                "confidence": 0.8897
            }
        ]

    },


    "poori.jpg": {

        "yolo": [],

        "api1": [
            {
                "food": "Bhatura",
                "confidence": 0.9944
            },
            {
                "food": "Bhatura",
                "confidence": 0.9918
            },
            {
                "food": "CoconutChutney",
                "confidence": 0.9805
            },
            {
                "food": "Bhatura",
                "confidence": 0.9781
            }
        ],

        "api2": [
            {
                "food": "POORI",
                "confidence": 0.9167
            },
            {
                "food": "SAMBAR",
                "confidence": 0.8636
            }
        ]

    },


    "samosa.jpg": {

        "yolo": [
            {
                "food": "alooparatha",
                "confidence": 0.7348
            }
        ],

        "api1": [
            {
                "food": "GreenChutney",
                "confidence": 0.9941
            },
            {
                "food": "Samosa",
                "confidence": 0.9897
            },
            {
                "food": "Samosa",
                "confidence": 0.9888
            },
            {
                "food": "Samosa",
                "confidence": 0.9532
            }
        ],

        "api2": [
            {
                "food": "GREEN CHUTNEY",
                "confidence": 0.8344
            }
        ]

    },


    "voda.jpg": {

        "yolo": [
            {
                "food": "chickentikka",
                "confidence": 0.5628
            }
        ],

        "api1": [],

        "api2": [
            {
                "food": "VADA",
                "confidence": 0.9082
            },
            {
                "food": "WHITE CHUTNEY",
                "confidence": 0.5098
            }
        ]

    }

}


# ------------------------------------------------------------
# RUN TESTS
# ------------------------------------------------------------

print("=" * 70)

print(
    "       NUTRIVISION AI - HYBRID ENGINE V8 TEST"
)

print("=" * 70)


for image_name, data in TEST_CASES.items():

    print()
    print("=" * 70)

    print(
        f"IMAGE: {image_name}"
    )

    print("=" * 70)

    result = hybrid_decision(

        data["yolo"],

        data["api1"],

        data["api2"]

    )

    print()

    print("V8 FINAL RESULT")

    print("-" * 70)

    print(
        f"Food: {result['food']}"
    )

    print(
        f"Confidence: {result['confidence']} %"
    )

    print(
        f"Supporting Models: "
        f"{result['supporting_models']}"
    )

    print(
        f"Winner Margin: "
        f"{result['winner_margin']}"
    )

    print()

    print("Scores:")

    for food, score in result["scores"].items():

        print(
            f"  {food}: {score:.4f}"
        )


print()

print("=" * 70)

print(
    "              V8 TEST COMPLETED"
)

print("=" * 70)