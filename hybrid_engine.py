# ============================================================
# NUTRIVISION AI - FINAL HYBRID ENGINE V6
# ============================================================

from collections import defaultdict


# ============================================================
# FOOD NORMALIZATION
# ============================================================

ALIASES = {
    "chapathi": "chapati",
    "chapati": "chapati",

    "dosa": "dosa",

    "idli": "idli",

    "biryani": "biryani",

    "poha": "poha",

    "poori": "poori",
    "puri": "poori",

    "bhatura": "bhatura",

    "vada": "vada",

    "samosa": "samosa",

    "plainrice": "plainrice",
    "plain rice": "plainrice",
    "whiterice": "whiterice",
    "white rice": "whiterice",
    "rice": "rice",

    "chole": "chole",
    "pickle": "pickle",
    "papadam": "papadam",
    "curd": "curd",

    "coconutchutney": "coconutchutney",
    "coconut chutney": "coconutchutney",

    "greenchutney": "greenchutney",
    "green chutney": "greenchutney",

    "whitechutney": "whitechutney",
    "white chutney": "whitechutney",

    "redchutney": "redchutney",
    "red chutney": "redchutney",

    "sambar": "sambar",
    "curry": "curry",

    "omelette": "omelette",
    "egg": "egg",

    "rasgulla": "rasgulla",

    "chickentikka": "chickentikka",

    "kheer": "kheer",

    "alooparatha": "alooparatha",
}


# ============================================================
# FOOD CATEGORIES
# ============================================================

MAIN_FOODS = {
    "chapati",
    "dosa",
    "idli",
    "biryani",
    "poha",
    "poori",
    "bhatura",
    "vada",
    "samosa",
    "plainrice",
    "whiterice",
    "rice",
    "omelette",
    "alooparatha",
    "chole",
    "chickentikka",
}

ACCOMPANIMENTS = {
    "coconutchutney",
    "greenchutney",
    "whitechutney",
    "redchutney",
    "sambar",
    "curry",
    "pickle",
    "papadam",
    "curd",
}

SECONDARY_FOODS = {
    "egg",
    "kheer",
    "rasgulla",
}


# ============================================================
# NORMALIZE FOOD NAME
# ============================================================

def normalize_food(food):
    if food is None:
        return None

    food = str(food).strip().lower()

    food = " ".join(food.split())

    if food in ALIASES:
        return ALIASES[food]

    food_no_space = food.replace(" ", "")

    return ALIASES.get(food_no_space, food_no_space)


# ============================================================
# NORMALIZE PREDICTIONS
# ============================================================

def normalize_predictions(predictions):
    normalized = []

    if not predictions:
        return normalized

    for item in predictions:

        if not isinstance(item, dict):
            continue

        food = normalize_food(item.get("food"))

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if food is None:
            continue

        confidence = max(0.0, min(1.0, confidence))

        normalized.append({
            "food": food,
            "confidence": confidence
        })

    return normalized


# ============================================================
# AGGREGATE MODEL PREDICTIONS
# ============================================================

def aggregate_predictions(predictions):
    scores = defaultdict(float)

    for item in predictions:

        food = item["food"]
        confidence = item["confidence"]

        # Multiple detections of the same food should help,
        # but not grow without limit.
        scores[food] += confidence

    return dict(scores)


# ============================================================
# TOP SCORE
# ============================================================

def get_top_score(scores):

    if not scores:
        return None, 0.0

    food = max(scores, key=scores.get)

    return food, scores[food]


# ============================================================
# SUPPORTING MODEL COUNT
# ============================================================

def calculate_supporting_models(food, yolo, api1, api2):

    count = 0

    for predictions in (yolo, api1, api2):

        foods = {
            item["food"]
            for item in predictions
        }

        if food in foods:
            count += 1

    return count


# ============================================================
# MAIN-FOOD BONUS
# ============================================================

def main_food_bonus(food):

    if food in MAIN_FOODS:
        return 0.10

    return 0.0


# ============================================================
# ACCOMPANIMENT PENALTY
# ============================================================

def accompaniment_penalty(food):

    if food in ACCOMPANIMENTS:
        return 0.10

    return 0.0


# ============================================================
# POORI / BHATURA SPECIAL RULE
# ============================================================

def resolve_poori_bhatura(scores, api1, api2):

    poori_api2 = 0.0
    bhatura_api1 = 0.0

    for item in api2:
        if item["food"] == "poori":
            poori_api2 = max(poori_api2, item["confidence"])

    for item in api1:
        if item["food"] == "bhatura":
            bhatura_api1 += item["confidence"]

    # Strong explicit POORI evidence from API2 should win
    # over repeated Bhatura detections from API1.
    if poori_api2 >= 0.85:

        scores["poori"] = scores.get("poori", 0.0) + 0.70

        scores["bhatura"] = max(
            0.0,
            scores.get("bhatura", 0.0) - 0.50
        )

        return "poori"

    # If both are present but POORI has meaningful evidence,
    # prefer POORI because they are visually similar foods.
    if poori_api2 >= 0.70:

        poori_score = scores.get("poori", 0.0)
        bhatura_score = scores.get("bhatura", 0.0)

        if poori_score >= bhatura_score * 0.65:

            scores["poori"] = poori_score + 0.45

            scores["bhatura"] = max(
                0.0,
                bhatura_score - 0.30
            )

            return "poori"

    return None


# ============================================================
# PLAIN RICE / RICE SPECIAL RULE
# ============================================================

def resolve_rice(scores, yolo, api1, api2):

    plainrice_yolo = 0.0
    rice_api2 = 0.0
    whiterice_api1 = 0.0

    for item in yolo:
        if item["food"] == "plainrice":
            plainrice_yolo = max(
                plainrice_yolo,
                item["confidence"]
            )

    for item in api2:
        if item["food"] == "rice":
            rice_api2 = max(
                rice_api2,
                item["confidence"]
            )

    for item in api1:
        if item["food"] == "whiterice":
            whiterice_api1 = max(
                whiterice_api1,
                item["confidence"]
            )

    # Explicit YOLO plainrice detection is highly valuable.
    if plainrice_yolo >= 0.80:

        scores["plainrice"] = (
            scores.get("plainrice", 0.0)
            + 0.85
        )

        # Generic rice should not override explicit plainrice.
        scores["rice"] = max(
            0.0,
            scores.get("rice", 0.0) - 0.20
        )

        return "plainrice"

    # Combination of explicit plain/white rice evidence.
    if plainrice_yolo >= 0.60 and (
        rice_api2 >= 0.75 or
        whiterice_api1 >= 0.75
    ):

        scores["plainrice"] = (
            scores.get("plainrice", 0.0)
            + 0.60
        )

        return "plainrice"

    # Strong generic rice evidence should remain rice.
    if rice_api2 >= 0.85:

        scores["rice"] = (
            scores.get("rice", 0.0)
            + 0.20
        )

    return None


# ============================================================
# MEAL CONTEXT RULE
# ============================================================

def apply_meal_context(scores, yolo, api1, api2):

    # If this is clearly a meal containing several dishes,
    # do not allow a repeated component such as Bhatura to
    # automatically become the final image-level answer.

    unique_foods = set(scores.keys())

    meal_indicators = {
        "chole",
        "plainrice",
        "rice",
        "chapati",
        "bhatura",
        "pickle",
        "papadam",
        "curd",
        "sambar",
    }

    indicator_count = len(
        unique_foods.intersection(meal_indicators)
    )

    if indicator_count >= 4:

        # Protect explicit plainrice.
        if scores.get("plainrice", 0.0) > 0:

            scores["plainrice"] += 0.45

            scores["bhatura"] = max(
                0.0,
                scores.get("bhatura", 0.0) - 0.15
            )

        # Generic rice receives less priority than explicit
        # plainrice in a mixed meal.
        if "plainrice" in scores and "rice" in scores:

            if scores["plainrice"] >= scores["rice"]:

                scores["rice"] *= 0.85


# ============================================================
# SCORE ADJUSTMENT
# ============================================================

def adjust_scores(scores):

    adjusted = {}

    for food, score in scores.items():

        value = score

        value += main_food_bonus(food)

        value -= accompaniment_penalty(food)

        value = max(0.0, value)

        adjusted[food] = value

    return adjusted


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(scores, winner, supporting_models):

    if not scores or winner is None:
        return 0.0

    ordered = sorted(
        scores.values(),
        reverse=True
    )

    winner_score = ordered[0]

    second_score = (
        ordered[1]
        if len(ordered) > 1
        else 0.0
    )

    total = sum(scores.values())

    if total <= 0:
        return 0.0

    share = winner_score / total

    margin = winner_score - second_score

    # Score-share contribution.
    confidence = share * 100.0

    # Strong model agreement gives additional confidence.
    if supporting_models >= 2:
        confidence += 8.0

    if supporting_models >= 3:
        confidence += 5.0

    # Strong margin.
    if margin >= 1.0:
        confidence += 5.0
    elif margin >= 0.50:
        confidence += 2.0

    # Avoid unrealistic confidence for very close decisions.
    if margin < 0.10:
        confidence = min(confidence, 49.0)

    elif margin < 0.25:
        confidence = min(confidence, 65.0)

    confidence = max(
        0.0,
        min(100.0, confidence)
    )

    return round(confidence, 2)


# ============================================================
# DECISION LABEL
# ============================================================

def decision_label(confidence, margin):

    if margin < 0.10:
        return "AMBIGUOUS"

    if confidence >= 75:
        return "HIGH CONFIDENCE"

    if confidence >= 50:
        return "MODERATE CONFIDENCE"

    return "LOW CONFIDENCE"


# ============================================================
# DECISION REASON
# ============================================================

def get_decision_reason(
    winner,
    supporting_models,
    special_rule=None
):

    if special_rule == "poori":
        return (
            "Explicit POORI evidence preserved over "
            "Bhatura ambiguity."
        )

    if special_rule == "plainrice":
        return (
            "Explicit plain-rice evidence preserved "
            "over generic rice or meal-component detection."
        )

    if winner in {
        "dosa",
        "samosa",
        "poori",
        "bhatura",
    }:
        return (
            "Main-food preference applied over "
            "accompaniment or visually similar food."
        )

    if supporting_models >= 2:
        return (
            "Multi-model evidence supports the selected food."
        )

    return (
        "Context-aware single-model food selection."
    )


# ============================================================
# FINAL HYBRID DECISION
# ============================================================

def hybrid_decision(yolo_predictions, api1_predictions, api2_predictions):

    # --------------------------------------------------------
    # Normalize all model predictions
    # --------------------------------------------------------

    yolo = normalize_predictions(yolo_predictions)
    api1 = normalize_predictions(api1_predictions)
    api2 = normalize_predictions(api2_predictions)

    # --------------------------------------------------------
    # Aggregate evidence
    # --------------------------------------------------------

    yolo_scores = aggregate_predictions(yolo)
    api1_scores = aggregate_predictions(api1)
    api2_scores = aggregate_predictions(api2)

    scores = defaultdict(float)

    # --------------------------------------------------------
    # Model weights
    #
    # YOLO:
    # object/localization evidence
    #
    # API1:
    # food recognition evidence
    #
    # API2:
    # independent recognition evidence
    # --------------------------------------------------------

    for food, score in yolo_scores.items():
        scores[food] += score * 1.00

    for food, score in api1_scores.items():
        scores[food] += score * 0.85

    for food, score in api2_scores.items():
        scores[food] += score * 0.95

    # --------------------------------------------------------
    # Base score adjustments
    # --------------------------------------------------------

    scores = defaultdict(
        float,
        adjust_scores(dict(scores))
    )

    # --------------------------------------------------------
    # Special food-pair rules
    # --------------------------------------------------------

    special_rule = None

    poori_result = resolve_poori_bhatura(
        scores,
        api1,
        api2
    )

    if poori_result == "poori":
        special_rule = "poori"

    rice_result = resolve_rice(
        scores,
        yolo,
        api1,
        api2
    )

    if rice_result == "plainrice":
        special_rule = "plainrice"

    # --------------------------------------------------------
    # Meal context
    # --------------------------------------------------------

    apply_meal_context(
        scores,
        yolo,
        api1,
        api2
    )

    # --------------------------------------------------------
    # Samosa protection
    # --------------------------------------------------------

    if scores.get("samosa", 0.0) > 0:

        samosa_score = scores["samosa"]
        chutney_score = scores.get(
            "greenchutney",
            0.0
        )

        if samosa_score > 0:

            scores["samosa"] += 0.20

            if chutney_score > 0:

                scores["greenchutney"] = max(
                    0.0,
                    chutney_score - 0.15
                )

    # --------------------------------------------------------
    # Dosa protection
    # --------------------------------------------------------

    if scores.get("dosa", 0.0) > 0:

        dosa_score = scores["dosa"]

        accompaniment_total = sum(
            scores.get(food, 0.0)
            for food in {
                "coconutchutney",
                "sambar",
                "whitechutney",
                "redchutney",
                "curry",
            }
        )

        # If Dosa is present, accompaniment detections
        # should not collectively overpower it.
        if accompaniment_total > dosa_score:

            scores["dosa"] += 0.20

    # --------------------------------------------------------
    # Final winner
    # --------------------------------------------------------

    final_scores = dict(scores)

    winner, winner_score = get_top_score(
        final_scores
    )

    # --------------------------------------------------------
    # Supporting model count
    # --------------------------------------------------------

    supporting_models = calculate_supporting_models(
        winner,
        yolo,
        api1,
        api2
    )

    # --------------------------------------------------------
    # Winner margin
    # --------------------------------------------------------

    ordered_scores = sorted(
        final_scores.values(),
        reverse=True
    )

    second_score = (
        ordered_scores[1]
        if len(ordered_scores) > 1
        else 0.0
    )

    winner_margin = winner_score - second_score

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        final_scores,
        winner,
        supporting_models
    )

    decision = decision_label(
        confidence,
        winner_margin
    )

    reason = get_decision_reason(
        winner,
        supporting_models,
        special_rule
    )

    # --------------------------------------------------------
    # Rounded output
    # --------------------------------------------------------

    rounded_scores = {
        food: round(score, 4)
        for food, score in final_scores.items()
    }

    return {
        "food": winner,
        "confidence": confidence,
        "decision": decision,
        "supporting_models": supporting_models,
        "winner_margin": round(
            winner_margin,
            4
        ),
        "decision_reason": reason,
        "scores": rounded_scores,
    }


# ============================================================
# END OF HYBRID ENGINE V6
# ============================================================