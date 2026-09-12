# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V5
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
# Side dishes
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
# Normalize food names
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

        "aloo paratha": "alooparatha",

        "plain rice": "plainrice",
        "whiterice": "plainrice",

        "rice": "plainrice",

        "coconutchutney": "coconutchutney",
        "coconut chutney": "coconutchutney",

        "greenchutney": "greenchutney",
        "green chutney": "greenchutney",

        "whitechutney": "whitechutney",
        "white chutney": "whitechutney",

        "redchutney": "redchutney",

        "papadam": "papadam",

        "curd": "curd",
    }

    return aliases.get(name, name)


# ------------------------------------------------------------
# Food type
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
    "api1": 0.90,
    "api2": 1.00,
}


# ------------------------------------------------------------
# API 2 strong foods
# ------------------------------------------------------------

API2_SPECIAL_FOODS = {

    "dosa",
    "idli",
    "poori",
    "vada",
    "chapati",

}


# ------------------------------------------------------------
# Known confusing food pairs
# ------------------------------------------------------------

CONFUSION_PAIRS = {

    ("poori", "bhatura"),
    ("dosa", "omelette"),
    ("idli", "rasgulla"),
    ("samosa", "alooparatha"),
    ("plainrice", "rice"),

}


# ------------------------------------------------------------
# Calculate model evidence
# ------------------------------------------------------------

def calculate_model_evidence(model_name, results):

    evidence = {}

    if not results:
        return evidence

    model_weight = MODEL_WEIGHTS[model_name]

    # Keep only meaningful predictions
    predictions = []

    for prediction in results:

        confidence = float(
            prediction["confidence"]
        )

        if confidence < 0.30:
            continue

        food = normalize_food_name(
            prediction["food"]
        )

        predictions.append(
            (food, confidence)
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Multiple detections of same food from one model
    # should NOT be counted fully.
    # --------------------------------------------------------

    food_confidences = {}

    for food, confidence in predictions:

        if food not in food_confidences:
            food_confidences[food] = []

        food_confidences[food].append(
            confidence
        )

    for food, confidences in food_confidences.items():

        # Highest confidence is the main evidence
        best_confidence = max(confidences)

        score = (
            best_confidence *
            model_weight
        )

        # Small bonus for a second independent
        # detection of the same food.
        if len(confidences) >= 2:

            second_best = sorted(
                confidences,
                reverse=True
            )[1]

            score += second_best * 0.20

        # Side dishes are less important
        if food_type(food) == "side":

            score *= 0.70

        # API 2 special food bonus
        if (
            model_name == "api2"
            and food in API2_SPECIAL_FOODS
        ):

            score *= 1.20

        evidence[food] = score

    return evidence


# ------------------------------------------------------------
# Hybrid decision V5
# ------------------------------------------------------------

def hybrid_decision_v5(
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

    model_evidence = {}

    # --------------------------------------------------------
    # Calculate individual model evidence
    # --------------------------------------------------------

    for model_name, results in all_results.items():

        evidence = calculate_model_evidence(
            model_name,
            results
        )

        model_evidence[model_name] = evidence

        for food, score in evidence.items():

            if food not in scores:
                scores[food] = 0.0

            scores[food] += score

    # --------------------------------------------------------
    # Agreement bonus
    # --------------------------------------------------------

    supporting_models = {}

    for food in scores:

        count = 0

        for model_name in model_evidence:

            if food in model_evidence[model_name]:
                count += 1

        supporting_models[food] = count

        # Two independent models agree
        if count >= 2:

            scores[food] += 0.35

        # All three agree
        if count == 3:

            scores[food] += 0.50

    # --------------------------------------------------------
    # Special rule:
    #
    # If API 2 strongly detects a food and another model
    # produces a confusing result, prefer API 2.
    # --------------------------------------------------------

    if "api2" in model_evidence:

        for food, api2_score in model_evidence["api2"].items():

            if food not in MAIN_FOODS:
                continue

            if api2_score < 0.75:
                continue

            for other_food in list(scores.keys()):

                if other_food == food:
                    continue

                pair = (
                    food,
                    other_food
                )

                reverse_pair = (
                    other_food,
                    food
                )

                if (
                    pair in CONFUSION_PAIRS
                    or reverse_pair in CONFUSION_PAIRS
                ):

                    # Reduce confusing competitor
                    scores[other_food] *= 0.75

                    # Give API 2 food a small reliability bonus
                    scores[food] += 0.20

    # --------------------------------------------------------
    # Remove extremely weak side-dish-only candidates
    # when a main food exists.
    # --------------------------------------------------------

    main_foods = [
        food
        for food in scores
        if food_type(food) == "main"
    ]

    if main_foods:

        for food in list(scores.keys()):

            if food_type(food) == "side":

                # Don't completely remove it,
                # but reduce its influence.
                scores[food] *= 0.80

    # --------------------------------------------------------
    # No result
    # --------------------------------------------------------

    if not scores:

        return {

            "food": "unknown",
            "confidence": 0.0,
            "scores": {},
            "supporting_models": 0,
            "winner_margin": 0.0,

        }

    # --------------------------------------------------------
    # Sort candidates
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_food = ranked[0][0]
    best_score = ranked[0][1]

    if len(ranked) > 1:

        second_score = ranked[1][1]

    else:

        second_score = 0.0

    winner_margin = (
        best_score - second_score
    )

    # --------------------------------------------------------
    # Supporting models
    # --------------------------------------------------------

    best_support = supporting_models.get(
        best_food,
        0
    )

    # --------------------------------------------------------
    # Confidence calculation
    #
    # This is a hybrid confidence score,
    # NOT a calibrated probability.
    # --------------------------------------------------------

    base_confidence = best_score * 32

    # Agreement improves confidence
    if best_support == 2:

        base_confidence += 12

    elif best_support == 3:

        base_confidence += 20

    # Strong winner margin improves confidence
    if winner_margin >= 0.75:

        base_confidence += 10

    elif winner_margin >= 0.50:

        base_confidence += 6

    elif winner_margin >= 0.25:

        base_confidence += 3

    # Penalize ambiguous decisions
    if winner_margin < 0.20:

        base_confidence -= 8

    confidence = max(
        0,
        min(
            base_confidence,
            100
        )
    )

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
            for food, score in ranked
        },

        "supporting_models": best_support,

        "winner_margin": round(
            winner_margin,
            4
        ),

    }