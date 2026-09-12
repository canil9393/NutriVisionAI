# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V10
# ============================================================

import re


# ============================================================
# MAIN FOOD CLASSES
# ============================================================

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


# ============================================================
# SIDE DISHES
# ============================================================

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


# ============================================================
# NORMALIZATION
# ============================================================

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

        "plainrice": "plainrice",
        "whiterice": "plainrice",
        "whiterice": "plainrice",

        "aloo paratha": "alooparatha",
        "alooparatha": "alooparatha",

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


# ============================================================
# FOOD TYPE
# ============================================================

def food_type(food):

    food = normalize_food_name(food)

    if food in SIDE_DISHES:
        return "side"

    return "main"


# ============================================================
# MODEL WEIGHTS
# ============================================================

MODEL_WEIGHTS = {

    "yolo": 1.00,
    "api1": 0.90,
    "api2": 1.00,

}


# ============================================================
# API 2 SPECIAL FOODS
# ============================================================

API2_SPECIAL_FOODS = {

    "dosa",
    "idli",
    "poori",
    "vada",
    "chapati",

}


# ============================================================
# V10 FOOD PRIORITY
# ============================================================

# These relationships are important because some foods
# are commonly confused by individual models.

FOOD_RELATIONSHIPS = {

    "poori": {
        "bhatura": 0.65
    },

    "bhatura": {
        "poori": 0.65
    },

    "dosa": {
        "omelette": 0.60
    },

    "omelette": {
        "dosa": 0.60
    },

    "samosa": {
        "alooparatha": 0.55
    },

    "alooparatha": {
        "samosa": 0.55
    },

}


# ============================================================
# PARAMETERS
# ============================================================

MIN_CONFIDENCE = 0.30

SIDE_PENALTY = 0.50

REPEATED_DETECTION_DECAY = 0.35

TWO_MODEL_BONUS = 0.45

THREE_MODEL_BONUS = 0.75

STRONG_AGREEMENT_BONUS = 0.25

API2_SPECIAL_MULTIPLIER = 1.20


# ============================================================
# HELPER
# ============================================================

def add_prediction_score(
    scores,
    model_name,
    prediction,
    occurrence_index
):

    food = normalize_food_name(
        prediction.get("food", "")
    )

    if not food:
        return None, 0.0

    confidence = float(
        prediction.get("confidence", 0.0)
    )

    if confidence < MIN_CONFIDENCE:
        return food, 0.0

    model_weight = MODEL_WEIGHTS[model_name]

    score = confidence * model_weight

    # --------------------------------------------------------
    # Repeated detections
    #
    # First detection gets full weight.
    # Additional detections receive reduced weight.
    # This prevents 3 duplicate detections from completely
    # overpowering another model.
    # --------------------------------------------------------

    if occurrence_index > 0:

        score *= (
            REPEATED_DETECTION_DECAY
            ** occurrence_index
        )

    # --------------------------------------------------------
    # Side dish penalty
    # --------------------------------------------------------

    if food_type(food) == "side":

        score *= SIDE_PENALTY

    # --------------------------------------------------------
    # API 2 special-food advantage
    # --------------------------------------------------------

    if (

        model_name == "api2"

        and food in API2_SPECIAL_FOODS

    ):

        score *= API2_SPECIAL_MULTIPLIER

    return food, score


# ============================================================
# MAIN HYBRID DECISION
# ============================================================

def hybrid_decision(

    yolo_results,
    api1_results,
    api2_results

):

    scores = {}

    model_foods = {

        "yolo": set(),
        "api1": set(),
        "api2": set(),

    }

    model_max_conf = {

        "yolo": {},
        "api1": {},
        "api2": {},

    }

    all_results = [

        ("yolo", yolo_results),
        ("api1", api1_results),
        ("api2", api2_results),

    ]

    # ========================================================
    # STEP 1 - SCORE ALL PREDICTIONS
    # ========================================================

    for model_name, results in all_results:

        if not results:
            continue

        occurrences = {}

        for prediction in results:

            food = normalize_food_name(
                prediction.get("food", "")
            )

            occurrence_index = occurrences.get(
                food,
                0
            )

            occurrences[food] = occurrence_index + 1

            food, score = add_prediction_score(

                scores,
                model_name,
                prediction,
                occurrence_index

            )

            if not food:
                continue

            confidence = float(
                prediction.get("confidence", 0.0)
            )

            if confidence < MIN_CONFIDENCE:
                continue

            model_foods[
                model_name
            ].add(food)

            previous = model_max_conf[
                model_name
            ].get(food, 0.0)

            if confidence > previous:

                model_max_conf[
                    model_name
                ][food] = confidence

            scores[food] = (
                scores.get(food, 0.0)
                + score
            )

    # ========================================================
    # STEP 2 - MODEL AGREEMENT
    # ========================================================

    for food in list(scores.keys()):

        supporting_models = sum(

            1

            for model_name in model_foods

            if food in model_foods[model_name]

        )

        # Two models
        if supporting_models >= 2:

            scores[food] += TWO_MODEL_BONUS

        # Three models
        if supporting_models == 3:

            scores[food] += THREE_MODEL_BONUS

        # Strong agreement
        strong_models = sum(

            1

            for model_name in model_foods

            if model_max_conf[
                model_name
            ].get(food, 0.0) >= 0.75

        )

        if strong_models >= 2:

            scores[food] += STRONG_AGREEMENT_BONUS

    # ========================================================
    # STEP 3 - SIDE DISH SUPPRESSION
    # ========================================================

    main_foods = [

        food

        for food in scores

        if food_type(food) == "main"

    ]

    if main_foods:

        for food in list(scores.keys()):

            if food_type(food) == "side":

                scores[food] *= 0.85

    # ========================================================
    # STEP 4 - FOOD RELATIONSHIP HANDLING
    # ========================================================

    for food, relationships in FOOD_RELATIONSHIPS.items():

        if food not in scores:
            continue

        for competitor, factor in relationships.items():

            if competitor not in scores:
                continue

            food_support = sum(

                1

                for model_name in model_foods

                if food in model_foods[model_name]

            )

            competitor_support = sum(

                1

                for model_name in model_foods

                if competitor in model_foods[model_name]

            )

            # If the food has stronger model support,
            # suppress the competitor.
            if food_support > competitor_support:

                scores[competitor] *= factor

    # ========================================================
    # STEP 5 - SPECIAL SAMOSA PROTECTION
    # ========================================================

    # Samosa is frequently accompanied by green chutney.
    # If a model identifies BOTH samosa and green chutney,
    # the main food should receive priority.

    if "samosa" in scores:

        if "greenchutney" in scores:

            samosa_support = sum(

                1

                for model_name in model_foods

                if "samosa"
                in model_foods[model_name]

            )

            chutney_support = sum(

                1

                for model_name in model_foods

                if "greenchutney"
                in model_foods[model_name]

            )

            if samosa_support >= chutney_support:

                scores["greenchutney"] *= 0.50

    # ========================================================
    # STEP 6 - EMPTY RESULT
    # ========================================================

    scores = {

        food: score

        for food, score in scores.items()

        if score > 0

    }

    if not scores:

        return {

            "food": "unknown",
            "confidence": 0.0,
            "scores": {},
            "supporting_models": 0,
            "winner_margin": 0.0,

        }

    # ========================================================
    # STEP 7 - RANK
    # ========================================================

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

        else 0.0

    )

    winner_margin = (

        best_score - second_score
    )

    # ========================================================
    # STEP 8 - SUPPORTING MODELS
    # ========================================================

    supporting_models = sum(

        1

        for model_name in model_foods

        if best_food in model_foods[model_name]

    )

    # ========================================================
    # STEP 9 - CONFIDENCE
    # ========================================================

    score_component = min(

        best_score / 2.5,
        1.0

    )

    margin_component = min(

        winner_margin / 1.5,
        1.0

    )

    support_component = {

        1: 0.45,
        2: 0.80,
        3: 1.00,

    }.get(

        supporting_models,
        0.40

    )

    confidence = (

        score_component * 0.40

        + margin_component * 0.30

        + support_component * 0.30

    ) * 100

    # Single-model results should not appear
    # artificially overconfident.

    if supporting_models == 1:

        confidence *= 0.85

    confidence = max(

        0.0,
        min(confidence, 100.0)

    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

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

        "supporting_models":
            supporting_models,

        "winner_margin":
            round(
                winner_margin,
                4
            ),

    }