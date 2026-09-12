# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V7
# Conflict-Aware + Context-Aware Decision Engine
# ============================================================

import re


# ------------------------------------------------------------
# MAIN FOODS
# ------------------------------------------------------------

MAIN_FOODS = {
    "dosa", "idli", "poori", "vada", "samosa", "chapati",
    "biryani", "poha", "khichdi", "plainrice", "alooparatha",
    "palakpaneer", "chickentikka", "rasgulla", "omelette",
    "bhatura", "chole", "kheer", "egg"
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
    "curd"
}


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_food_name(name):

    name = name.lower().strip()

    name = re.sub(
        r"[\s_-]+",
        "",
        name
    )

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
        "greenchutney": "greenchutney",
        "whitechutney": "whitechutney",
        "redchutney": "redchutney",

        "alooparatha": "alooparatha",
    }

    return aliases.get(
        name,
        name
    )


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

    # API 1 is useful but produces several false positives
    "api1": 0.80,

    # API 2 performs well on several South Indian foods
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
# CLOSE FOOD CONFLICTS
#
# These foods are visually similar and therefore require
# special conflict handling.
# ------------------------------------------------------------

CONFLICT_GROUPS = [

    {
        "poori",
        "bhatura",
    },

    {
        "dosa",
        "idli",
    },

    {
        "chapati",
        "alooparatha",
    },

    {
        "plainrice",
        "biryani",
    },

]


# ------------------------------------------------------------
# FIND CONFLICT GROUP
# ------------------------------------------------------------

def get_conflict_group(food):

    food = normalize_food_name(food)

    for group in CONFLICT_GROUPS:

        if food in group:
            return group

    return set()


# ------------------------------------------------------------
# MAIN HYBRID DECISION
# ------------------------------------------------------------

def hybrid_decision(
    yolo_results,
    api1_results,
    api2_results
):

    scores = {}

    model_foods = {}

    # --------------------------------------------------------
    # ALL MODEL RESULTS
    # --------------------------------------------------------

    all_results = [

        ("yolo", yolo_results),

        ("api1", api1_results),

        ("api2", api2_results),

    ]


    # --------------------------------------------------------
    # STEP 1
    # COLLECT SCORES
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

            # Ignore extremely weak detections
            if confidence < 0.30:
                continue

            model_foods[
                model_name
            ].add(food)

            weight = MODEL_WEIGHTS[
                model_name
            ]

            score = (
                confidence * weight
            )


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


            scores[food] = (
                scores.get(food, 0)
                + score
            )


    # --------------------------------------------------------
    # STEP 2
    # MODEL AGREEMENT
    # --------------------------------------------------------

    for food in list(scores.keys()):

        supporting_models = sum(

            food in foods

            for foods in model_foods.values()

        )


        # Two models agree
        if supporting_models >= 2:

            scores[food] += 0.45


        # Three models agree
        if supporting_models == 3:

            scores[food] += 0.50


    # --------------------------------------------------------
    # STEP 3
    # MAIN FOOD PRIORITY
    # --------------------------------------------------------

    for food in list(scores.keys()):

        if food_type(food) == "main":

            scores[food] *= 1.10


    # --------------------------------------------------------
    # STEP 4
    # SUPPRESS SIDES WHEN MAIN FOOD EXISTS
    # --------------------------------------------------------

    main_scores = [

        score

        for food, score in scores.items()

        if food_type(food) == "main"

    ]


    if main_scores:

        strongest_main = max(
            main_scores
        )

        for food in list(scores.keys()):

            if food_type(food) == "side":

                if (
                    scores[food]
                    < strongest_main * 0.75
                ):

                    scores[food] *= 0.50


    # --------------------------------------------------------
    # STEP 5
    # CONFLICT RESOLUTION
    # --------------------------------------------------------

    for group in CONFLICT_GROUPS:

        candidates = [

            food

            for food in group

            if food in scores

        ]

        if len(candidates) < 2:
            continue


        # Sort by current score
        candidates.sort(
            key=lambda food: scores[food],
            reverse=True
        )

        first = candidates[0]
        second = candidates[1]

        first_score = scores[first]
        second_score = scores[second]


        # ----------------------------------------------------
        # Check API 2 support
        # ----------------------------------------------------

        api2_supported = {

            food

            for food in model_foods["api2"]

            if food in group

        }


        # ----------------------------------------------------
        # If API 2 identifies one candidate specifically,
        # give that candidate additional authority.
        # ----------------------------------------------------

        if len(api2_supported) == 1:

            preferred = next(
                iter(api2_supported)
            )

            scores[preferred] *= 1.18


        # ----------------------------------------------------
        # Repeated API 1 detections should NOT automatically
        # dominate a specialized API 2 prediction.
        # ----------------------------------------------------

        # Re-sort after API 2 adjustment

        candidates.sort(
            key=lambda food: scores[food],
            reverse=True
        )


    # --------------------------------------------------------
    # STEP 6
    # RECALCULATE WINNER
    # --------------------------------------------------------

    ranked = sorted(

        scores.items(),

        key=lambda item: item[1],

        reverse=True

    )


    if not ranked:

        return {

            "food": "unknown",

            "confidence": 0.0,

            "supporting_models": 0,

            "winner_margin": 0.0,

            "scores": {}

        }


    best_food = ranked[0][0]

    best_score = ranked[0][1]


    second_score = (

        ranked[1][1]

        if len(ranked) > 1

        else 0.0

    )


    winner_margin = (

        best_score
        - second_score

    )


    # --------------------------------------------------------
    # STEP 7
    # SUPPORTING MODELS
    # --------------------------------------------------------

    supporting_models = sum(

        best_food in foods

        for foods in model_foods.values()

    )


    # --------------------------------------------------------
    # STEP 8
    # CONFLICT FLAG
    # --------------------------------------------------------

    conflict_group = get_conflict_group(
        best_food
    )

    conflict_present = any(

        food in scores

        for food in conflict_group

        if food != best_food

    )


    # --------------------------------------------------------
    # STEP 9
    # CONFIDENCE
    #
    # Confidence is based on:
    #
    #   - winner score
    #   - winner margin
    #   - model agreement
    #   - conflict
    #
    # It is NOT a calibrated probability.
    # --------------------------------------------------------

    base_confidence = min(

        best_score * 32,

        72

    )


    margin_bonus = min(

        winner_margin * 18,

        18

    )


    agreement_bonus = {

        1: 0,

        2: 10,

        3: 16,

    }.get(

        supporting_models,

        0

    )


    confidence = (

        base_confidence

        + margin_bonus

        + agreement_bonus

    )


    # --------------------------------------------------------
    # PENALIZE UNRESOLVED CONFLICT
    # --------------------------------------------------------

    if conflict_present:

        confidence -= 8


    # --------------------------------------------------------
    # SINGLE-MODEL SAFETY
    #
    # A single model should not easily produce 90-100%.
    # --------------------------------------------------------

    if supporting_models == 1:

        confidence = min(

            confidence,

            82

        )


    confidence = max(

        0,

        min(

            confidence,

            100

        )

    )


    # --------------------------------------------------------
    # UNCERTAINTY
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