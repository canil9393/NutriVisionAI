# ============================================================
# NUTRIVISION AI - HYBRID DECISION ENGINE V8
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

        "coconut chutney": "coconutchutney",
        "coconutchutney": "coconutchutney",

        "green chutney": "greenchutney",

        "white chutney": "whitechutney",

        "red chutney": "redchutney",

        "greenchutney": "greenchutney",

        "whitechutney": "whitechutney",

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
# ------------------------------------------------------------

API2_SPECIAL_FOODS = {

    "dosa",
    "idli",
    "poori",
    "vada",
    "chapati",

}


# ------------------------------------------------------------
# KNOWN CONFUSION PAIRS
#
# These rules are based on our previous testing.
# ------------------------------------------------------------

PAIRWISE_PRIORITY = {

    # API 2 performs better for Poori/Bhatura conflict
    ("poori", "bhatura"): "api2",
    ("bhatura", "poori"): "api2",

}


# ------------------------------------------------------------
# MODEL-SPECIFIC FOOD PRIORITY
# ------------------------------------------------------------

MODEL_FOOD_PRIORITY = {

    "api2": {

        "dosa": 1.20,
        "idli": 1.20,
        "poori": 1.35,
        "vada": 1.20,
        "chapati": 1.20,

    },

}


# ------------------------------------------------------------
# SIDE DISH PENALTY
# ------------------------------------------------------------

SIDE_DISH_WEIGHT = 0.50


# ------------------------------------------------------------
# AGREEMENT BONUS
# ------------------------------------------------------------

TWO_MODEL_BONUS = 0.40

THREE_MODEL_BONUS = 0.60


# ------------------------------------------------------------
# MAIN FOOD BONUS
#
# Helps main foods beat accompanying chutneys.
# ------------------------------------------------------------

MAIN_FOOD_BONUS = 0.10


# ------------------------------------------------------------
# GET MODEL SUPPORT
# ------------------------------------------------------------

def get_model_support(food, model_results):

    support = []

    for model_name, results in model_results:

        for prediction in results:

            predicted_food = normalize_food_name(
                prediction["food"]
            )

            confidence = float(
                prediction["confidence"]
            )

            if predicted_food == food and confidence >= 0.30:

                support.append(
                    (
                        model_name,
                        confidence
                    )
                )

                break

    return support


# ------------------------------------------------------------
# CALCULATE BASE SCORES
# ------------------------------------------------------------

def calculate_scores(
    yolo_results,
    api1_results,
    api2_results
):

    scores = {}

    model_results = [

        ("yolo", yolo_results),

        ("api1", api1_results),

        ("api2", api2_results),

    ]

    for model_name, results in model_results:

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

            # Ignore weak predictions
            if confidence < 0.30:
                continue

            score = confidence * base_weight

            # ------------------------------------------------
            # Side dish penalty
            # ------------------------------------------------

            if food_type(food) == "side":

                score *= SIDE_DISH_WEIGHT

            else:

                score += MAIN_FOOD_BONUS

            # ------------------------------------------------
            # Model-specific priority
            # ------------------------------------------------

            priority = MODEL_FOOD_PRIORITY.get(
                model_name,
                {}
            ).get(
                food,
                1.0
            )

            score *= priority

            # ------------------------------------------------
            # Add score
            # ------------------------------------------------

            if food not in scores:

                scores[food] = 0.0

            scores[food] += score

    return scores


# ------------------------------------------------------------
# AGREEMENT BONUS
# ------------------------------------------------------------

def apply_agreement_bonus(
    scores,
    yolo_results,
    api1_results,
    api2_results
):

    model_results = [

        ("yolo", yolo_results),

        ("api1", api1_results),

        ("api2", api2_results),

    ]

    for food in list(scores.keys()):

        supporting_models = get_model_support(
            food,
            model_results
        )

        count = len(supporting_models)

        if count >= 2:

            scores[food] += TWO_MODEL_BONUS

        if count == 3:

            scores[food] += THREE_MODEL_BONUS

    return scores


# ------------------------------------------------------------
# PAIRWISE CONFLICT RESOLUTION
# ------------------------------------------------------------

def apply_pairwise_resolution(
    scores,
    yolo_results,
    api1_results,
    api2_results
):

    if not scores:
        return scores

    model_results = [

        ("yolo", yolo_results),

        ("api1", api1_results),

        ("api2", api2_results),

    ]

    foods = list(scores.keys())

    for i in range(len(foods)):

        for j in range(i + 1, len(foods)):

            food_a = foods[i]

            food_b = foods[j]

            pair = (food_a, food_b)

            reverse_pair = (food_b, food_a)

            preferred_model = (
                PAIRWISE_PRIORITY.get(pair)
                or
                PAIRWISE_PRIORITY.get(reverse_pair)
            )

            if not preferred_model:
                continue

            # ------------------------------------------------
            # Find confidence from preferred model
            # ------------------------------------------------

            preferred_support = 0.0

            for model_name, results in model_results:

                if model_name != preferred_model:
                    continue

                for prediction in results:

                    food = normalize_food_name(
                        prediction["food"]
                    )

                    if food in (food_a, food_b):

                        confidence = float(
                            prediction["confidence"]
                        )

                        if confidence > preferred_support:

                            preferred_support = confidence

            if preferred_support >= 0.70:

                preferred_food = None

                for model_name, results in model_results:

                    if model_name != preferred_model:
                        continue

                    for prediction in results:

                        food = normalize_food_name(
                            prediction["food"]
                        )

                        if food in (food_a, food_b):

                            confidence = float(
                                prediction["confidence"]
                            )

                            if confidence == preferred_support:

                                preferred_food = food

                if preferred_food:

                    scores[preferred_food] += 0.35

    return scores


# ------------------------------------------------------------
# REMOVE DOMINANT SIDE DISHES
#
# If a main food has reasonable evidence, do not allow
# chutney/sambar to become the final answer simply because
# several side-dish predictions exist.
# ------------------------------------------------------------

def protect_main_food(scores):

    main_food_scores = {

        food: score

        for food, score in scores.items()

        if food_type(food) == "main"

    }

    if not main_food_scores:

        return scores

    best_main_score = max(
        main_food_scores.values()
    )

    # Side dishes should not dominate a reasonable main food.
    for food in list(scores.keys()):

        if food_type(food) == "side":

            if scores[food] > best_main_score:

                scores[food] = (
                    best_main_score * 0.70
                )

    return scores


# ------------------------------------------------------------
# FINAL CONFIDENCE
# ------------------------------------------------------------

def calculate_confidence(
    best_food,
    best_score,
    second_score,
    supporting_models
):

    if best_score <= 0:

        return 0.0

    # --------------------------------------------------------
    # Base confidence
    # --------------------------------------------------------

    confidence = 45.0

    # Score strength
    confidence += min(
        best_score * 12.0,
        25.0
    )

    # Winner margin
    margin = best_score - second_score

    confidence += min(
        margin * 18.0,
        20.0
    )

    # Model agreement
    if supporting_models >= 3:

        confidence += 15.0

    elif supporting_models == 2:

        confidence += 10.0

    # --------------------------------------------------------
    # Low-support penalty
    # --------------------------------------------------------

    if supporting_models == 1:

        confidence -= 15.0

    # --------------------------------------------------------
    # Keep confidence in range
    # --------------------------------------------------------

    confidence = max(
        0.0,
        min(confidence, 100.0)
    )

    return round(
        confidence,
        2
    )


# ------------------------------------------------------------
# MAIN HYBRID DECISION
# ------------------------------------------------------------

def hybrid_decision(
    yolo_results,
    api1_results,
    api2_results
):

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    scores = calculate_scores(
        yolo_results,
        api1_results,
        api2_results
    )

    # --------------------------------------------------------
    # No detections
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
    # Agreement
    # --------------------------------------------------------

    scores = apply_agreement_bonus(

        scores,

        yolo_results,

        api1_results,

        api2_results

    )

    # --------------------------------------------------------
    # Pairwise conflict resolution
    # --------------------------------------------------------

    scores = apply_pairwise_resolution(

        scores,

        yolo_results,

        api1_results,

        api2_results

    )

    # --------------------------------------------------------
    # Protect main foods
    # --------------------------------------------------------

    scores = protect_main_food(
        scores
    )

    # --------------------------------------------------------
    # Sort scores
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

    # --------------------------------------------------------
    # Supporting models
    # --------------------------------------------------------

    model_results = [

        ("yolo", yolo_results),

        ("api1", api1_results),

        ("api2", api2_results),

    ]

    supporting_models = len(
        get_model_support(
            best_food,
            model_results
        )
    )

    # --------------------------------------------------------
    # Winner margin
    # --------------------------------------------------------

    winner_margin = (
        best_score - second_score
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(

        best_food,

        best_score,

        second_score,

        supporting_models

    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "food": best_food,

        "confidence": confidence,

        "supporting_models": supporting_models,

        "winner_margin": round(
            winner_margin,
            4
        ),

        "scores": {

            food: round(
                score,
                4
            )

            for food, score in ranked

        }

    }