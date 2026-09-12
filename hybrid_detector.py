# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V2
# ============================================================

import re


# ------------------------------------------------------------
# Main food classes
# ------------------------------------------------------------

MAIN_FOODS = {
    "dosa",
    "idli",
    "poori",
    "vada",
    "samosa",
    "chapati",
    "biryani",
    "poha",
    "khichdi",
    "plainrice",
    "alooparatha",
    "palakpaneer",
    "chickentikka",
    "rasgulla",
    "omelette",
    "bhatura",
    "chole",
    "kheer",
    "egg",
}


# ------------------------------------------------------------
# Side dishes / accompaniments
# ------------------------------------------------------------

SIDE_DISHES = {
    "coconutchutney",
    "greenchutney",
    "whitechutney",
    "redchutney",
    "sambar",
    "pickle",
    "curry",
}


# ------------------------------------------------------------
# Food name normalization
# ------------------------------------------------------------

def normalize_food_name(name):

    name = name.lower().strip()

    # Remove spaces, -, _
    name = re.sub(r"[\s_-]+", "", name)

    aliases = {

        "chapathi": "chapati",
        "chappathi": "chapati",

        "dosai": "dosa",

        "idly": "idli",

        "vadai": "vada",

        "puri": "poori",

        "aloo paratha": "alooparatha",
        "alooparatha": "alooparatha",

        "plain rice": "plainrice",
        "whiterice": "plainrice",

        "chickentikka": "chickentikka",

        "coconutchutney": "coconutchutney",
        "coconut chutney": "coconutchutney",

        "greenchutney": "greenchutney",
        "green chutney": "greenchutney",

        "whitechutney": "whitechutney",
        "white chutney": "whitechutney",

        "redchutney": "redchutney",

    }

    return aliases.get(name, name)


# ------------------------------------------------------------
# Determine food type
# ------------------------------------------------------------

def food_type(food):

    food = normalize_food_name(food)

    if food in SIDE_DISHES:
        return "side"

    return "main"


# ------------------------------------------------------------
# Model weights
# ------------------------------------------------------------

MODEL_WEIGHTS = {

    "yolo": 1.00,
    "api1": 0.75,
    "api2": 1.00,
}


# ------------------------------------------------------------
# Food-specific model priority
#
# API 2 performs particularly well on:
# Dosa, Idli, Poori, Vada
# ------------------------------------------------------------

API2_SPECIAL_FOODS = {
     "dosa",
    "idli",
    "poori",
    "vada",
    "chapati",
    "samosa",
}


# ------------------------------------------------------------
# Hybrid decision
# ------------------------------------------------------------

def hybrid_decision(yolo_results, api1_results, api2_results):

    scores = {}

    all_results = [

        ("yolo", yolo_results),
        ("api1", api1_results),
        ("api2", api2_results),

    ]

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    for model_name, results in all_results:

        if not results:
            continue

        model_weight = MODEL_WEIGHTS[model_name]

        for prediction in results:

            food = normalize_food_name(
                prediction["food"]
            )

            confidence = float(
                prediction["confidence"]
            )

            # Ignore very weak predictions
            if confidence < 0.30:
                continue

            score = confidence * model_weight

            # ------------------------------------------------
            # Side dishes receive a penalty
            # ------------------------------------------------

            if food_type(food) == "side":
                score *= 0.50

            # ------------------------------------------------
            # API 2 gets additional priority for foods
            # where our testing showed strong performance
            # ------------------------------------------------

            if (
                model_name == "api2"
                and food in API2_SPECIAL_FOODS
            ):
                score *= 1.20

            if food not in scores:
                scores[food] = 0.0

            scores[food] += score

    # --------------------------------------------------------
    # Agreement bonus
    # --------------------------------------------------------

    model_foods = {}

    for model_name, results in all_results:

        foods = set()

        for prediction in results:

            food = normalize_food_name(
                prediction["food"]
            )

            if prediction["confidence"] >= 0.30:
                foods.add(food)

        model_foods[model_name] = foods

    all_foods = set(scores.keys())

    for food in all_foods:

        models_detecting_food = 0

        for foods in model_foods.values():

            if food in foods:
                models_detecting_food += 1

        # Two models agree
        if models_detecting_food >= 2:
            scores[food] += 0.40

        # All three agree
        if models_detecting_food == 3:
            scores[food] += 0.50

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if not scores:

        return {
            "food": "unknown",
            "confidence": 0.0,
            "scores": {}
        }

    # --------------------------------------------------------
    # Select highest score
    # --------------------------------------------------------

    best_food = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_food]

    # --------------------------------------------------------
    # Convert score into a readable confidence
    #
    # This is a hybrid confidence score, NOT a calibrated
    # probability.
    # --------------------------------------------------------

    confidence = min(
        best_score * 50,
        100
    )

    return {

        "food": best_food,

        "confidence": round(
            confidence,
            2
        ),

        "scores": {
            food: round(score, 4)
            for food, score in scores.items()
        }

    }