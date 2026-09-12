# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V9
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
# FOOD NORMALIZATION
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

        "aloo paratha": "alooparatha",
        "aloo_paratha": "alooparatha",

        "plain rice": "plainrice",
        "whiterice": "plainrice",
        "white rice": "plainrice",

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

        "papad": "papadam",

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
#
# API 2 is particularly useful for these foods.
# ============================================================

API2_SPECIAL_FOODS = {

    "dosa",
    "idli",
    "poori",
    "vada",
    "chapati",

}


# ============================================================
# CONFUSING FOOD PAIRS
#
# These are visually similar foods where one model may
# frequently confuse the other.
# ============================================================

CONFUSION_PAIRS = {

    frozenset({"poori", "bhatura"}),
    frozenset({"dosa", "omelette"}),
    frozenset({"vada", "chickentikka"}),
    frozenset({"chapati", "alooparatha"}),

}


# ============================================================
# SPECIAL PAIR DECISION
# ============================================================

def apply_special_pair_rules(scores, model_predictions):

    # --------------------------------------------------------
    # POORI vs BHATURA
    #
    # Important for poori.jpg:
    #
    # API 1:
    #   Bhatura 99%
    #
    # API 2:
    #   Poori 91%
    #
    # If API 2 strongly identifies Poori and YOLO has no
    # useful main-food detection, prefer Poori.
    # --------------------------------------------------------

    poori_api2 = model_predictions.get("api2", {}).get("poori", 0.0)
    bhatura_api1 = model_predictions.get("api1", {}).get("bhatura", 0.0)

    yolo_poori = model_predictions.get("yolo", {}).get("poori", 0.0)
    yolo_bhatura = model_predictions.get("yolo", {}).get("bhatura", 0.0)

    if poori_api2 >= 0.75:

        # Strong API2 Poori evidence
        scores["poori"] = scores.get("poori", 0.0) + 0.45

        # Reduce API1-only Bhatura dominance
        if bhatura_api1 >= 0.75 and yolo_bhatura < 0.50:

            scores["bhatura"] = scores.get("bhatura", 0.0) * 0.65

    # --------------------------------------------------------
    # DOSA vs OMELETTE
    # --------------------------------------------------------

    dosa_api2 = model_predictions.get("api2", {}).get("dosa", 0.0)
    omelette_yolo = model_predictions.get("yolo", {}).get("omelette", 0.0)

    if dosa_api2 >= 0.80:

        scores["dosa"] = scores.get("dosa", 0.0) + 0.30

        if omelette_yolo > 0:
            scores["omelette"] = scores.get(
                "omelette", 0.0
            ) * 0.85

    # --------------------------------------------------------
    # VADA vs CHICKEN TIKKA
    # --------------------------------------------------------

    vada_api2 = model_predictions.get("api2", {}).get("vada", 0.0)
    chicken_yolo = model_predictions.get(
        "yolo", {}
    ).get("chickentikka", 0.0)

    if vada_api2 >= 0.80:

        scores["vada"] = scores.get("vada", 0.0) + 0.30

        if chicken_yolo > 0:
            scores["chickentikka"] = scores.get(
                "chickentikka", 0.0
            ) * 0.85

    return scores


# ============================================================
# HYBRID DECISION V9
# ============================================================

def hybrid_decision(
    yolo_results,
    api1_results,
    api2_results
):

    all_results = [

        ("yolo", yolo_results or []),
        ("api1", api1_results or []),
        ("api2", api2_results or []),

    ]


    # ========================================================
    # STEP 1
    # Keep the BEST prediction of each food PER MODEL
    #
    # This prevents:
    #
    # Samosa 98%
    # Samosa 98%
    # Samosa 95%
    #
    # from artificially becoming ~2.9 points.
    # ========================================================

    model_predictions = {

        "yolo": {},
        "api1": {},
        "api2": {},

    }


    for model_name, results in all_results:

        for prediction in results:

            if not isinstance(prediction, dict):
                continue

            if "food" not in prediction:
                continue

            if "confidence" not in prediction:
                continue

            food = normalize_food_name(
                prediction["food"]
            )

            try:
                confidence = float(
                    prediction["confidence"]
                )
            except:

                continue


            # Support both:
            # 0.91
            # 91
            #
            # Internally use 0-1.

            if confidence > 1.0:

                confidence = confidence / 100.0


            if confidence < 0.30:
                continue


            old_confidence = model_predictions[
                model_name
            ].get(food, 0.0)


            # Keep highest confidence for same food
            model_predictions[
                model_name
            ][food] = max(
                old_confidence,
                confidence
            )


    # ========================================================
    # STEP 2
    # Calculate base scores
    # ========================================================

    scores = {}


    for model_name in [
        "yolo",
        "api1",
        "api2"
    ]:

        predictions = model_predictions[model_name]

        model_weight = MODEL_WEIGHTS[
            model_name
        ]


        for food, confidence in predictions.items():

            score = confidence * model_weight


            # ------------------------------------------------
            # SIDE DISH PENALTY
            # ------------------------------------------------

            if food_type(food) == "side":

                score *= 0.50


            # ------------------------------------------------
            # API 2 SPECIAL FOOD BOOST
            # ------------------------------------------------

            if (
                model_name == "api2"
                and food in API2_SPECIAL_FOODS
            ):

                score *= 1.25


            if food not in scores:

                scores[food] = 0.0


            scores[food] += score


    # ========================================================
    # STEP 3
    # Model agreement
    # ========================================================

    for food in list(scores.keys()):

        supporting_models = 0


        for model_name in [
            "yolo",
            "api1",
            "api2"
        ]:

            if food in model_predictions[
                model_name
            ]:

                supporting_models += 1


        # -----------------------------------------------
        # Two-model agreement
        # -----------------------------------------------

        if supporting_models >= 2:

            scores[food] += 0.40


        # -----------------------------------------------
        # Three-model agreement
        # -----------------------------------------------

        if supporting_models == 3:

            scores[food] += 0.45


    # ========================================================
    # STEP 4
    # Strong API 2 evidence
    # ========================================================

    for food in API2_SPECIAL_FOODS:

        api2_conf = model_predictions[
            "api2"
        ].get(food, 0.0)


        if api2_conf >= 0.85:

            if food not in scores:

                scores[food] = 0.0


            scores[food] += 0.20


    # ========================================================
    # STEP 5
    # Special confusion rules
    # ========================================================

    scores = apply_special_pair_rules(
        scores,
        model_predictions
    )


    # ========================================================
    # STEP 6
    # No result
    # ========================================================

    if not scores:

        return {

            "food": "unknown",
            "confidence": 0.0,
            "scores": {},
            "supporting_models": 0,
            "winner_margin": 0.0,

        }


    # ========================================================
    # STEP 7
    # Sort candidates
    # ========================================================

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


    # ========================================================
    # STEP 8
    # Supporting models
    # ========================================================

    supporting_models = 0


    for model_name in [
        "yolo",
        "api1",
        "api2"
    ]:

        if best_food in model_predictions[
            model_name
        ]:

            supporting_models += 1


    # ========================================================
    # STEP 9
    # Confidence calculation
    #
    # Confidence combines:
    #
    # 1. Absolute evidence
    # 2. Winner margin
    # 3. Model agreement
    #
    # It is NOT a calibrated probability.
    # ========================================================

    # Absolute score contribution
    evidence_score = min(
        best_score / 2.2,
        1.0
    )


    # Margin contribution
    margin_score = min(
        winner_margin / 1.5,
        1.0
    )


    # Model support
    support_score = {

        1: 0.45,
        2: 0.78,
        3: 1.00,

    }.get(
        supporting_models,
        0.30
    )


    # Weighted confidence
    raw_confidence = (

        evidence_score * 0.35
        +
        margin_score * 0.35
        +
        support_score * 0.30

    )


    confidence = raw_confidence * 100


    # ========================================================
    # STEP 10
    # Confidence safety rules
    # ========================================================

    # If only one model supports the winner and the margin
    # is small, do not report extremely high confidence.

    if supporting_models == 1:

        if winner_margin < 0.40:

            confidence = min(
                confidence,
                55
            )

        elif winner_margin < 0.70:

            confidence = min(
                confidence,
                70
            )

        elif winner_margin < 1.00:

            confidence = min(
                confidence,
                82
            )


    # If two models agree, allow high confidence only when
    # there is a reasonable margin.

    if supporting_models == 2:

        if winner_margin < 0.50:

            confidence = min(
                confidence,
                75
            )


    # Three models agreeing is strong evidence.

    if supporting_models == 3:

        confidence = min(
            confidence + 5,
            100
        )


    # Never allow negative or >100
    confidence = max(
        0,
        min(confidence, 100)
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

            for food, score in ranked

        },

    }