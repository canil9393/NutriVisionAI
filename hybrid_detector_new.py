# ============================================================
# NUTRIVISION AI
# MODULE 3D - HYBRID CONFLICT RESOLUTION
# ============================================================

from collections import defaultdict


# Model weights
YOLO_WEIGHT = 1.0
API1_WEIGHT = 0.8
API2_WEIGHT = 1.2

# Agreement bonus
AGREEMENT_BONUS = 0.30

# Minimum confidence required for a prediction
MIN_CONFIDENCE = 0.30


def normalize_food_name(food):

    food = food.lower().strip()

    aliases = {
        "chapathi": "chapati",
        "chappathi": "chapati",

        "white rice": "plainrice",
        "white_rice": "plainrice",

        "coconut chutney": "coconutchutney",
        "coconutchutney": "coconutchutney",

        "green chutney": "greenchutney",
        "greenchutney": "greenchutney",

        "white chutney": "whitechutney",
        "whitechutney": "whitechutney",

        "red chutney": "redchutney",
        "redchutney": "redchutney",

        "chicken tikka": "chickentikka",
        "chickentikka": "chickentikka",

        "poori": "poori",

        "dosa": "dosa",

        "idli": "idli",

        "vada": "vada",

        "samosa": "samosa",

        "poha": "poha",

        "biryani": "biryani",

        "kheer": "kheer",

        "bhatura": "bhatura",

        "omelette": "omelette",

        "egg": "egg",

        "rice": "rice",

        "pickle": "pickle",

        "papadam": "papadam",

        "curd": "curd",

        "sambar": "sambar",

        "curry": "curry",
    }

    return aliases.get(food, food)


def add_score(scores, prediction, weight):

    food = normalize_food_name(
        prediction["food"]
    )

    confidence = float(
        prediction["confidence"]
    )

    if confidence < MIN_CONFIDENCE:
        return

    scores[food] += confidence * weight


def hybrid_decision(
    yolo_results,
    api1_results,
    api2_results
):

    scores = defaultdict(float)

    # --------------------------------------------------------
    # STEP 1 — Add weighted model scores
    # --------------------------------------------------------

    for prediction in yolo_results:
        add_score(
            scores,
            prediction,
            YOLO_WEIGHT
        )

    for prediction in api1_results:
        add_score(
            scores,
            prediction,
            API1_WEIGHT
        )

    for prediction in api2_results:
        add_score(
            scores,
            prediction,
            API2_WEIGHT
        )

    # --------------------------------------------------------
    # STEP 2 — Find model-level food predictions
    # --------------------------------------------------------

    yolo_foods = set(
        normalize_food_name(
            p["food"]
        )
        for p in yolo_results
    )

    api1_foods = set(
        normalize_food_name(
            p["food"]
        )
        for p in api1_results
    )

    api2_foods = set(
        normalize_food_name(
            p["food"]
        )
        for p in api2_results
    )

    # --------------------------------------------------------
    # STEP 3 — Agreement bonus
    # --------------------------------------------------------

    for food in scores:

        agreement_count = 0

        if food in yolo_foods:
            agreement_count += 1

        if food in api1_foods:
            agreement_count += 1

        if food in api2_foods:
            agreement_count += 1

        if agreement_count >= 2:

            scores[food] += (
                AGREEMENT_BONUS *
                (agreement_count - 1)
            )

    # --------------------------------------------------------
    # STEP 4 — Select highest score
    # --------------------------------------------------------

    if not scores:

        return {
            "food": "unknown",
            "confidence": 0,
            "scores": {}
        }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_food = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    # --------------------------------------------------------
    # STEP 5 — Convert score to relative confidence
    # --------------------------------------------------------

    total_score = sum(
        scores.values()
    )

    if total_score > 0:

        confidence = (
            best_score /
            total_score
        ) * 100

    else:

        confidence = 0

    return {
        "food": best_food,
        "confidence": round(
            confidence,
            2
        ),
        "scores": dict(scores)
    }