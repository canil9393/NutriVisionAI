# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V4
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
# NORMALIZATION
# ------------------------------------------------------------

def normalize_food_name(name):

    name = str(name).lower().strip()

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

        "rice": "plainrice",

        "chickentikka": "chickentikka",

        "coconutchutney": "coconutchutney",
        "coconut chutney": "coconutchutney",

        "greenchutney": "greenchutney",
        "green chutney": "greenchutney",

        "whitechutney": "whitechutney",
        "white chutney": "whitechutney",

        "redchutney": "redchutney",

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
    "api1": 0.85,
    "api2": 1.00,

}


# ------------------------------------------------------------
# API 2 SPECIAL FOODS
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
# API 1 SPECIAL FOODS
# ------------------------------------------------------------

API1_SPECIAL_FOODS = {

    "biryani",
    "poha",
    "idli",
    "samosa",
}


# ------------------------------------------------------------
# MAIN FOOD PRIORITY
# ------------------------------------------------------------

def is_main_food(food):

    return normalize_food_name(food) in MAIN_FOODS


# ------------------------------------------------------------
# HYBRID DECISION V4
# ------------------------------------------------------------

def hybrid_decision_v4(
    yolo_results,
    api1_results,
    api2_results
):

    scores = {}

    # --------------------------------------------------------
    # Store model results
    # --------------------------------------------------------

    all_results = [

        ("yolo", yolo_results),
        ("api1", api1_results),
        ("api2", api2_results),

    ]


    # --------------------------------------------------------
    # STEP 1
    # Calculate weighted scores
    # --------------------------------------------------------

    for model_name, results in all_results:

        if not results:
            continue

        base_weight = MODEL_WEIGHTS[model_name]

        for prediction in results:

            food = normalize_food_name(
                prediction["food"]
            )

            confidence = float(
                prediction["confidence"]
            )

            # Ignore very weak detections
            if confidence < 0.30:
                continue


            score = confidence * base_weight


            # ------------------------------------------------
            # Side dish penalty
            # ------------------------------------------------

            if food_type(food) == "side":

                score *= 0.55


            # ------------------------------------------------
            # API 2 special-food bonus
            # ------------------------------------------------

            if (
                model_name == "api2"
                and food in API2_SPECIAL_FOODS
            ):

                score *= 1.20


            # ------------------------------------------------
            # API 1 special-food bonus
            # ------------------------------------------------

            if (
                model_name == "api1"
                and food in API1_SPECIAL_FOODS
            ):

                score *= 1.10


            # ------------------------------------------------
            # Add score
            # ------------------------------------------------

            if food not in scores:

                scores[food] = 0.0

            scores[food] += score


    # --------------------------------------------------------
    # STEP 2
    # Collect foods detected by each model
    # --------------------------------------------------------

    model_foods = {}

    for model_name, results in all_results:

        foods = set()

        if results:

            for prediction in results:

                confidence = float(
                    prediction["confidence"]
                )

                if confidence < 0.30:
                    continue

                food = normalize_food_name(
                    prediction["food"]
                )

                foods.add(food)

        model_foods[model_name] = foods


    # --------------------------------------------------------
    # STEP 3
    # Agreement bonus
    # --------------------------------------------------------

    for food in list(scores.keys()):

        agreement_count = 0

        for foods in model_foods.values():

            if food in foods:

                agreement_count += 1


        # Two models agree
        if agreement_count >= 2:

            scores[food] += 0.35


        # Three models agree
        if agreement_count == 3:

            scores[food] += 0.50


    # --------------------------------------------------------
    # STEP 4
    # Main food vs side dish protection
    #
    # If a main food has reasonable evidence, prevent a
    # side dish from winning only because of high confidence.
    # --------------------------------------------------------

    main_food_scores = {

        food: score

        for food, score in scores.items()

        if is_main_food(food)

    }


    side_food_scores = {

        food: score

        for food, score in scores.items()

        if food_type(food) == "side"

    }

        # --------------------------------------------------------
    # STEP 4B
    # Poori vs Bhatura conflict resolution
    #
    # API 2 is specifically detecting Poori while API 1
    # is repeatedly detecting Bhatura.
    #
    # Give API 2 priority when Poori confidence is strong.
    # --------------------------------------------------------

    poori_api2_confidence = 0.0

    if api2_results:

        for prediction in api2_results:

            food = normalize_food_name(
                prediction["food"]
            )

            confidence = float(
                prediction["confidence"]
            )

            if food == "poori":

                poori_api2_confidence = max(
                    poori_api2_confidence,
                    confidence
                )


    if poori_api2_confidence >= 0.85:

        # Reduce Bhatura score when API 2 strongly identifies Poori

        if "bhatura" in scores:

            scores["bhatura"] *= 0.60


    # --------------------------------------------------------
    # STEP 5
    # Select winner
    # --------------------------------------------------------

    if not scores:

        return {

            "food": "unknown",

            "confidence": 0.0,

            "scores": {}

        }


    best_food = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_food]


    # --------------------------------------------------------
    # STEP 6
    # If best result is a side dish but a main food is close,
    # prefer the main food.
    # --------------------------------------------------------

    if (
        food_type(best_food) == "side"
        and main_food_scores
    ):

        best_main_food = max(
            main_food_scores,
            key=main_food_scores.get
        )

        best_main_score = main_food_scores[
            best_main_food
        ]


        # Main food is sufficiently close
        if best_main_score >= best_score * 0.80:

            best_food = best_main_food
            best_score = best_main_score


    # --------------------------------------------------------
    # STEP 7
    # Calculate second-best score
    # --------------------------------------------------------

    sorted_scores = sorted(
        scores.values(),
        reverse=True
    )


    if len(sorted_scores) >= 2:

        second_score = sorted_scores[1]

    else:

        second_score = 0.0


    # --------------------------------------------------------
    # STEP 8
    # Winner margin
    # --------------------------------------------------------

    if best_score > 0:

        margin = (
            best_score - second_score
        ) / best_score

    else:

        margin = 0.0


    # --------------------------------------------------------
    # STEP 9
    # Number of models supporting winner
    # --------------------------------------------------------

    supporting_models = 0

    for foods in model_foods.values():

        if best_food in foods:

            supporting_models += 1


    # --------------------------------------------------------
    # STEP 10
    # Confidence calculation
    #
    # This is NOT a calibrated probability.
    # It is a hybrid confidence indicator.
    # --------------------------------------------------------

    confidence = 0.0


    # Base evidence
    confidence += min(
        best_score * 30,
        45
    )


    # Model agreement
    if supporting_models == 2:

        confidence += 15

    elif supporting_models >= 3:

        confidence += 25


    # Winner margin
    confidence += min(
        margin * 30,
        25
    )


    # Strong API 2 special food
    if best_food in API2_SPECIAL_FOODS:

        if "api2" in model_foods:

            if best_food in model_foods["api2"]:

                confidence += 12


    # --------------------------------------------------------
    # Limit confidence
    # --------------------------------------------------------

    confidence = min(
        confidence,
        100
    )


    # --------------------------------------------------------
    # Low-confidence protection
    # --------------------------------------------------------

    if confidence < 25:

        final_food = "unknown"

    else:

        final_food = best_food


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "food": final_food,

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

        "supporting_models":
            supporting_models,

        "margin":
            round(
                margin,
                4
            ),

    }