# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V3
# ============================================================

import re


# ------------------------------------------------------------
# MAIN FOOD CLASSES
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
# SIDE DISHES
# ------------------------------------------------------------

SIDE_DISHES = {
    "coconutchutney",
    "greenchutney",
    "whitechutney",
    "redchutney",
    "sambar",
    "pickle",
    "curry",
    "papadam",
    "curd",
}


# ------------------------------------------------------------
# API / MODEL NAME NORMALIZATION
# ------------------------------------------------------------

def normalize_food_name(name):

    name = name.lower().strip()

    # Remove spaces, hyphens and underscores
    name = re.sub(r"[\s_-]+", "", name)

    aliases = {

        "chapathi": "chapati",
        "chappathi": "chapati",

        "dosai": "dosa",

        "idly": "idli",

        "vadai": "vada",

        "puri": "poori",

        "aloo paratha": "alooparatha",

        "plain rice": "plainrice",
        "whiterice": "plainrice",

        "coconutchutney": "coconutchutney",
        "coconut chutney": "coconutchutney",

        "greenchutney": "greenchutney",
        "green chutney": "greenchutney",

        "whitechutney": "whitechutney",
        "white chutney": "whitechutney",

        "redchutney": "redchutney",

        "chickentikka": "chickentikka",

        "papadam": "papadam",

    }

    return aliases.get(name, name)


# ------------------------------------------------------------
# FOOD TYPE
# ------------------------------------------------------------

def food_type(food):

    food = normalize_food_name(food)

    if food in SIDE_DISHES:
        return "side"

    return "main"


# ------------------------------------------------------------
# MODEL WEIGHTS
# ------------------------------------------------------------

MODEL_WEIGHTS = {

    "yolo": 1.00,
    "api1": 0.90,
    "api2": 1.00,

}


# ------------------------------------------------------------
# API 2 SPECIAL FOODS
#
# Based on our testing:
# Dosa, Idli, Poori, Vada and Chapati
# ------------------------------------------------------------

API2_SPECIAL_FOODS = {

    "dosa",
    "idli",
    "poori",
    "vada",
    "chapati",

}


# ------------------------------------------------------------
# V3 PARAMETERS
# ------------------------------------------------------------

# Ignore predictions below this confidence
MIN_CONFIDENCE = 0.30


# Side dishes should not dominate main foods
SIDE_DISH_WEIGHT = 0.45


# Agreement bonuses
MAIN_AGREEMENT_BONUS = 0.30
SIDE_AGREEMENT_BONUS = 0.05


# API 2 priority
API2_SPECIAL_WEIGHT = 1.20


# ------------------------------------------------------------
# GET STRONGEST PREDICTION FROM EACH MODEL
# ------------------------------------------------------------

def get_best_predictions(results):

    best_predictions = {}

    if not results:
        return best_predictions

    for prediction in results:

        food = normalize_food_name(
            prediction["food"]
        )

        confidence = float(
            prediction["confidence"]
        )

        if confidence < MIN_CONFIDENCE:
            continue

        # Keep ONLY strongest prediction
        # for each food from this model
        if (
            food not in best_predictions
            or confidence > best_predictions[food]
        ):

            best_predictions[food] = confidence

    return best_predictions


# ------------------------------------------------------------
# HYBRID DECISION
# ------------------------------------------------------------

def hybrid_decision(
    yolo_results,
    api1_results,
    api2_results
):

    all_results = {

        "yolo": yolo_results,
        "api1": api1_results,
        "api2": api2_results,

    }


    scores = {}

    # Stores which models detected each food
    model_foods = {}


    # --------------------------------------------------------
    # STEP 1
    # Remove duplicate detections
    # --------------------------------------------------------

    for model_name, results in all_results.items():

        best_predictions = get_best_predictions(results)

        model_foods[model_name] = set(
            best_predictions.keys()
        )


        # ----------------------------------------------------
        # STEP 2
        # Calculate weighted scores
        # ----------------------------------------------------

        for food, confidence in best_predictions.items():

            score = (
                confidence
                * MODEL_WEIGHTS[model_name]
            )


            # ------------------------------------------------
            # Side dish penalty
            # ------------------------------------------------

            if food_type(food) == "side":

                score *= SIDE_DISH_WEIGHT


            # ------------------------------------------------
            # API 2 special food bonus
            # ------------------------------------------------

            if (
                model_name == "api2"
                and food in API2_SPECIAL_FOODS
            ):

                score *= API2_SPECIAL_WEIGHT


            if food not in scores:

                scores[food] = 0.0


            scores[food] += score


    # --------------------------------------------------------
    # STEP 3
    # Agreement bonus
    # --------------------------------------------------------

    for food in list(scores.keys()):

        detecting_models = 0

        for foods in model_foods.values():

            if food in foods:

                detecting_models += 1


        # Main food agreement
        if food_type(food) == "main":

            if detecting_models >= 2:

                scores[food] += (
                    MAIN_AGREEMENT_BONUS
                )


        # Side dish agreement
        else:

            if detecting_models >= 2:

                scores[food] += (
                    SIDE_AGREEMENT_BONUS
                )


    # --------------------------------------------------------
    # STEP 4
    # No valid detection
    # --------------------------------------------------------

    if not scores:

        return {

            "food": "unknown",

            "confidence": 0.0,

            "scores": {},

        }


    # --------------------------------------------------------
    # STEP 5
    # Select strongest food
    # --------------------------------------------------------

    best_food = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_food]


    # --------------------------------------------------------
    # STEP 6
    # Relative confidence
    # --------------------------------------------------------

    total_score = sum(
        scores.values()
    )


    if total_score > 0:

        confidence = (
            best_score
            / total_score
        ) * 100

    else:

        confidence = 0.0


    # --------------------------------------------------------
    # STEP 7
    # Return result
    # --------------------------------------------------------

    return {

        "food": best_food,

        "confidence": round(
            confidence,
            2
        ),

        "scores": {

            food: round(
                score,
                4
            )

            for food, score in scores.items()

        },

    }