# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V6
# Context-Aware Hybrid Decision Engine
# ============================================================

import re


# ------------------------------------------------------------
# FOOD CLASSES
# ------------------------------------------------------------

MAIN_FOODS = {
    "dosa", "idli", "poori", "vada", "samosa", "chapati",
    "biryani", "poha", "khichdi", "plainrice", "alooparatha",
    "palakpaneer", "chickentikka", "rasgulla", "omelette",
    "bhatura", "chole", "kheer", "egg"
}

SIDE_DISHES = {
    "coconutchutney", "greenchutney", "whitechutney",
    "redchutney", "sambar", "pickle", "curry",
    "papadam", "curd"
}


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_food_name(name):

    name = name.lower().strip()
    name = re.sub(r"[\s_-]+", "", name)

    aliases = {
        "chapathi": "chapati",
        "chappathi": "chapati",
        "dosai": "dosa",
        "idly": "idli",
        "vadai": "vada",
        "puri": "poori",

        "plainrice": "plainrice",
        "whiterice": "plainrice",
        "rice": "plainrice",

        "coconutchutney": "coconutchutney",
        "coconutchutney": "coconutchutney",

        "greenchutney": "greenchutney",
        "whitechutney": "whitechutney",
        "redchutney": "redchutney",

        "aloo paratha": "alooparatha",
        "alooparatha": "alooparatha",
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
    "api2": 1.10,
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
}


# ------------------------------------------------------------
# FOOD RELATIONSHIPS
# ------------------------------------------------------------

RELATED_FOODS = {

    "poori": {"bhatura"},
    "bhatura": {"poori"},

    "dosa": {"idli"},
    "idli": {"dosa"},

    "chapati": {"alooparatha"},
    "alooparatha": {"chapati"},

    "plainrice": {"biryani"},
    "biryani": {"plainrice"},

}


# ------------------------------------------------------------
# V6 DECISION ENGINE
# ------------------------------------------------------------

def hybrid_decision(yolo_results, api1_results, api2_results):

    scores = {}
    model_foods = {}

    all_results = [
        ("yolo", yolo_results),
        ("api1", api1_results),
        ("api2", api2_results),
    ]

    # --------------------------------------------------------
    # STEP 1: SCORE PREDICTIONS
    # --------------------------------------------------------

    for model_name, results in all_results:

        model_foods[model_name] = set()

        if not results:
            continue

        for prediction in results:

            food = normalize_food_name(
                prediction["food"]
            )

            confidence = float(
                prediction["confidence"]
            )

            if confidence < 0.30:
                continue

            model_foods[model_name].add(food)

            weight = MODEL_WEIGHTS[model_name]

            score = confidence * weight

            # ------------------------------------------------
            # SIDE DISH PENALTY
            # ------------------------------------------------

            if food_type(food) == "side":
                score *= 0.55

            # ------------------------------------------------
            # API 2 SPECIAL FOOD BOOST
            # ------------------------------------------------

            if (
                model_name == "api2"
                and food in API2_SPECIAL_FOODS
            ):
                score *= 1.20

            scores[food] = scores.get(food, 0) + score

    # --------------------------------------------------------
    # STEP 2: MODEL AGREEMENT
    # --------------------------------------------------------

    for food in list(scores.keys()):

        supporting_models = sum(
            food in foods
            for foods in model_foods.values()
        )

        # Two independent models agree
        if supporting_models >= 2:
            scores[food] += 0.45

        # All three agree
        if supporting_models == 3:
            scores[food] += 0.50

    # --------------------------------------------------------
    # STEP 3: MAIN FOOD PRIORITY
    # --------------------------------------------------------

    for food in list(scores.keys()):

        if food_type(food) == "main":

            scores[food] *= 1.10

    # --------------------------------------------------------
    # STEP 4: SIDE DISH SUPPRESSION
    #
    # If a strong main food exists, sides should normally
    # not become the final answer.
    # --------------------------------------------------------

    strongest_main = 0.0

    for food, score in scores.items():

        if food_type(food) == "main":
            strongest_main = max(
                strongest_main,
                score
            )

    if strongest_main > 0:

        for food in list(scores.keys()):

            if food_type(food) == "side":

                # Suppress side dishes when a strong
                # main food candidate exists.
                if scores[food] < strongest_main * 0.80:
                    scores[food] *= 0.50

    # --------------------------------------------------------
    # STEP 5: RELATED FOOD ADJUSTMENT
    #
    # Prevent closely related foods from dominating merely
    # because of one high-confidence API result.
    # --------------------------------------------------------

    for food in list(scores.keys()):

        for related in RELATED_FOODS.get(food, set()):

            if related in scores:

                difference = abs(
                    scores[food] - scores[related]
                )

                # Keep competition balanced.
                if difference < 0.35:

                    scores[food] *= 0.95

    # --------------------------------------------------------
    # NO RESULT
    # --------------------------------------------------------

    if not scores:

        return {
            "food": "unknown",
            "confidence": 0.0,
            "supporting_models": 0,
            "winner_margin": 0.0,
            "scores": {}
        }

    # --------------------------------------------------------
    # STEP 6: SELECT WINNER
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_food = ranked[0][0]
    best_score = ranked[0][1]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0
    )

    winner_margin = (
        best_score - second_score
    )

    # --------------------------------------------------------
    # SUPPORTING MODELS
    # --------------------------------------------------------

    supporting_models = sum(
        best_food in foods
        for foods in model_foods.values()
    )

    # --------------------------------------------------------
    # CONFIDENCE
    #
    # Confidence considers:
    #   1. Winner score
    #   2. Margin over second candidate
    #   3. Number of supporting models
    # --------------------------------------------------------

    base_confidence = min(
        best_score * 35,
        80
    )

    margin_bonus = min(
        winner_margin * 20,
        15
    )

    agreement_bonus = {
        1: 0,
        2: 8,
        3: 15
    }.get(
        supporting_models,
        0
    )

    confidence = (
        base_confidence
        + margin_bonus
        + agreement_bonus
    )

    confidence = min(
        confidence,
        100
    )

    # --------------------------------------------------------
    # LOW CONFIDENCE PROTECTION
    # --------------------------------------------------------

    if (
        confidence < 35
        and winner_margin < 0.30
    ):
        final_food = "uncertain"
    else:
        final_food = best_food

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "food": final_food,

        "confidence": round(
            confidence,
            2
        ),

        "supporting_models":
            supporting_models,

        "winner_margin":
            round(
                winner_margin,
                4
            ),

        "scores": {
            food: round(
                score,
                4
            )
            for food, score in scores.items()
        }

    }