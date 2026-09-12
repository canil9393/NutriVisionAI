# ============================================================
# NUTRIVISION AI - FINAL HYBRID ENGINE V4
# ============================================================
#
# V4 focuses on:
# 1. Multi-model score fusion
# 2. Main-food vs side-dish conflict resolution
# 3. Visually similar food resolution
# 4. Explicit plain-rice protection
# 5. Confidence based on final decision
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# Foods that are normally the MAIN food in an image.
MAIN_FOODS = {
    "chapati",
    "chapathi",
    "dosa",
    "idli",
    "biryani",
    "poha",
    "poori",
    "bhatura",
    "samosa",
    "vada",
    "plainrice",
    "rice",
    "whiterice",
    "white_rice",
    "omelette",
    "alooparatha",
}


# Foods that are normally SIDE DISHES / ACCOMPANIMENTS.
SIDE_DISHES = {
    "coconutchutney",
    "coconutchutney",
    "sambar",
    "whitechutney",
    "redchutney",
    "greenchutney",
    "pickle",
    "papadam",
    "curd",
    "curry",
}


# Similar-food pairs.
SIMILAR_FOOD_PAIRS = {
    frozenset({"poori", "bhatura"}),
    frozenset({"dosa", "omelette"}),
    frozenset({"plainrice", "rice"}),
    frozenset({"plainrice", "whiterice"}),
    frozenset({"samosa", "alooparatha"}),
}


# Explicit food-name normalization.
ALIASES = {
    "chapathi": "chapati",
    "chappathi": "chapati",

    "coconut chutney": "coconutchutney",
    "coconut_chutney": "coconutchutney",

    "white chutney": "whitechutney",
    "red chutney": "redchutney",
    "green chutney": "greenchutney",

    "white rice": "whiterice",
    "white_rice": "whiterice",

    "plain rice": "plainrice",
    "plain_rice": "plainrice",

    "aloo paratha": "alooparatha",
    "aloo_paratha": "alooparatha",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_food(food):
    """
    Normalize food names so predictions from different models
    can be compared correctly.
    """

    if food is None:
        return None

    food = str(food).strip().lower()

    food = " ".join(food.split())

    if food in ALIASES:
        return ALIASES[food]

    food = food.replace(" ", "")

    if food in ALIASES:
        return ALIASES[food]

    return food


# ============================================================
# ADD MODEL PREDICTIONS
# ============================================================

def add_predictions(scores, predictions, model_weight):
    """
    Add predictions from one model into the common score table.

    Multiple detections of the same food are accumulated using
    the strongest confidence plus a smaller contribution from
    additional detections.
    """

    if not predictions:
        return

    food_groups = {}

    for prediction in predictions:

        if not isinstance(prediction, dict):
            continue

        food = normalize_food(prediction.get("food"))
        confidence = prediction.get("confidence", 0.0)

        if not food:
            continue

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        confidence = max(0.0, min(1.0, confidence))

        food_groups.setdefault(food, []).append(confidence)

    for food, confidences in food_groups.items():

        confidences.sort(reverse=True)

        # Strongest detection receives full contribution.
        contribution = confidences[0]

        # Additional detections give smaller supporting evidence.
        if len(confidences) > 1:
            contribution += 0.25 * confidences[1]

        if len(confidences) > 2:
            contribution += 0.10 * confidences[2]

        contribution *= model_weight

        scores[food] = scores.get(food, 0.0) + contribution


# ============================================================
# MODEL SUPPORT COUNT
# ============================================================

def get_model_support(food, yolo, api1, api2):
    """
    Count how many independent models support a food.
    """

    food = normalize_food(food)

    support = 0

    for predictions in (yolo, api1, api2):

        found = False

        for prediction in predictions or []:

            if not isinstance(prediction, dict):
                continue

            predicted_food = normalize_food(
                prediction.get("food")
            )

            if predicted_food == food:
                found = True
                break

        if found:
            support += 1

    return support


# ============================================================
# RAW SCORE FUSION
# ============================================================

def build_scores(yolo, api1, api2):
    """
    Combine YOLO + API1 + API2 predictions.

    YOLO is useful for visual object detection.
    API1/API2 provide additional classification evidence.
    """

    scores = {}

    # YOLO
    add_predictions(
        scores,
        yolo,
        model_weight=1.00
    )

    # API 1
    add_predictions(
        scores,
        api1,
        model_weight=0.90
    )

    # API 2
    add_predictions(
        scores,
        api2,
        model_weight=0.90
    )

    return scores


# ============================================================
# SIDE-DISH PENALTY
# ============================================================

def apply_side_dish_penalty(scores):
    """
    Prevent a side dish from winning over a clearly detected
    main food.

    This is deliberately a moderate penalty rather than
    completely removing side dishes.
    """

    adjusted = dict(scores)

    main_food_scores = {
        food: score
        for food, score in scores.items()
        if food in MAIN_FOODS
    }

    side_food_scores = {
        food: score
        for food, score in scores.items()
        if food in SIDE_DISHES
    }

    if not main_food_scores or not side_food_scores:
        return adjusted

    strongest_main = max(main_food_scores.values())

    for side_food, side_score in side_food_scores.items():

        # Side dishes are reduced when a meaningful main food
        # is already present.
        if strongest_main > 0.50:

            penalty = min(
                side_score * 0.35,
                strongest_main * 0.30
            )

            adjusted[side_food] = max(
                0.0,
                side_score - penalty
            )

    return adjusted


# ============================================================
# MAIN FOOD CONFLICT RESOLUTION
# ============================================================

def resolve_main_food_conflicts(scores, original_scores):
    """
    Resolve known food-vs-food conflicts.

    The goal is NOT to blindly select the highest score.

    Instead, if two foods are close and one is a more appropriate
    main-food interpretation, preserve the main-food candidate.
    """

    adjusted = dict(scores)

    # --------------------------------------------------------
    # 1. DOSA vs OMELETTE
    # --------------------------------------------------------

    if "dosa" in original_scores and "omelette" in original_scores:

        dosa_score = original_scores["dosa"]
        omelette_score = original_scores["omelette"]

        # If dosa is reasonably close to omelette, prefer dosa.
        if dosa_score >= omelette_score * 0.90:

            bonus = omelette_score * 0.18

            adjusted["dosa"] = (
                adjusted.get("dosa", 0.0) + bonus
            )

    # --------------------------------------------------------
    # 2. POORI vs BHATURA
    # --------------------------------------------------------

    if "poori" in original_scores and "bhatura" in original_scores:

        poori_score = original_scores["poori"]
        bhatura_score = original_scores["bhatura"]

        # Poori receives preference when it is reasonably
        # close to bhatura.
        if poori_score >= bhatura_score * 0.72:

            bonus = bhatura_score * 0.22

            adjusted["poori"] = (
                adjusted.get("poori", 0.0) + bonus
            )

    # --------------------------------------------------------
    # 3. SAMOSA vs GREEN CHUTNEY
    # --------------------------------------------------------

    if "samosa" in original_scores and "greenchutney" in original_scores:

        samosa_score = original_scores["samosa"]
        chutney_score = original_scores["greenchutney"]

        # Green chutney is normally an accompaniment.
        # Prefer samosa when its evidence is reasonably strong.
        if samosa_score >= chutney_score * 0.62:

            bonus = chutney_score * 0.30

            adjusted["samosa"] = (
                adjusted.get("samosa", 0.0) + bonus
            )

            adjusted["greenchutney"] = (
                adjusted.get("greenchutney", 0.0)
                - chutney_score * 0.15
            )

            adjusted["greenchutney"] = max(
                0.0,
                adjusted["greenchutney"]
            )

    # --------------------------------------------------------
    # 4. PLAIN RICE vs GENERIC RICE
    # --------------------------------------------------------

    if "plainrice" in original_scores and "rice" in original_scores:

        plainrice_score = original_scores["plainrice"]
        rice_score = original_scores["rice"]

        if plainrice_score >= rice_score * 0.70:

            bonus = rice_score * 0.25

            adjusted["plainrice"] = (
                adjusted.get("plainrice", 0.0) + bonus
            )

    # --------------------------------------------------------
    # 5. PLAIN RICE vs WHITE RICE
    # --------------------------------------------------------

    if "plainrice" in original_scores and "whiterice" in original_scores:

        plainrice_score = original_scores["plainrice"]
        whiterice_score = original_scores["whiterice"]

        if plainrice_score >= whiterice_score * 0.70:

            bonus = whiterice_score * 0.20

            adjusted["plainrice"] = (
                adjusted.get("plainrice", 0.0) + bonus
            )

    return adjusted


# ============================================================
# SMALL MAIN-FOOD PRIOR
# ============================================================

def apply_main_food_prior(scores):
    """
    Give a small preference to main foods.

    This is intentionally small so that a genuinely strong
    side-dish prediction can still win when appropriate.
    """

    adjusted = dict(scores)

    for food in list(adjusted.keys()):

        if food in MAIN_FOODS:

            adjusted[food] *= 1.08

    return adjusted


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(
    winner_score,
    runner_score,
    supporting_models,
    number_of_candidates
):
    """
    Calculate confidence from:

    - Winner strength
    - Winner margin
    - Independent model support
    - Number of competing candidates
    """

    if winner_score <= 0:
        return 0.0

    margin = max(
        0.0,
        winner_score - runner_score
    )

    # Relative margin.
    relative_margin = margin / max(
        winner_score,
        0.0001
    )

    # Base confidence.
    confidence = 35.0

    # Winner strength.
    confidence += min(
        25.0,
        winner_score * 12.0
    )

    # Margin.
    confidence += min(
        30.0,
        relative_margin * 40.0
    )

    # Independent model support.
    if supporting_models >= 3:
        confidence += 12.0
    elif supporting_models == 2:
        confidence += 7.0
    elif supporting_models == 1:
        confidence -= 2.0

    # Many candidates make the decision harder.
    if number_of_candidates >= 7:
        confidence -= 5.0
    elif number_of_candidates >= 5:
        confidence -= 3.0

    confidence = max(
        0.0,
        min(100.0, confidence)
    )

    return round(confidence, 2)


# ============================================================
# DECISION LABEL
# ============================================================

def get_decision(confidence, margin):
    """
    Convert numerical confidence into a readable decision.
    """

    if confidence >= 80.0 and margin >= 0.80:
        return "HIGH CONFIDENCE"

    if confidence >= 60.0 and margin >= 0.40:
        return "MODERATE CONFIDENCE"

    if margin < 0.20:
        return "AMBIGUOUS"

    return "LOW CONFIDENCE"


# ============================================================
# DECISION REASON
# ============================================================

def get_decision_reason(
    winner,
    original_scores,
    adjusted_scores
):
    """
    Explain why the final winner was selected.
    """

    winner = normalize_food(winner)

    # Main food vs side dish.
    if winner in MAIN_FOODS:

        side_candidates = [
            food
            for food in original_scores
            if food in SIDE_DISHES
        ]

        if side_candidates:
            strongest_side = max(
                side_candidates,
                key=lambda food: original_scores[food]
            )

            if (
                original_scores.get(strongest_side, 0.0)
                >= original_scores.get(winner, 0.0) * 0.70
            ):
                return (
                    "Main-food preference applied over "
                    "accompaniment detection."
                )

    # Specific known conflicts.
    if winner == "dosa":

        if "omelette" in original_scores:
            return (
                "Dosa preserved through visual-conflict "
                "resolution."
            )

        return (
            "Dosa selected using fused model evidence."
        )

    if winner == "poori":

        if "bhatura" in original_scores:
            return (
                "Poori preserved over visually similar "
                "bhatura."
            )

    if winner == "samosa":

        if "greenchutney" in original_scores:
            return (
                "Main-food samosa preferred over "
                "green-chutney accompaniment."
            )

    if winner == "plainrice":

        if "rice" in original_scores:
            return (
                "Explicit plain-rice detection preserved "
                "over generic rice classification."
            )

    return (
        "Context-aware multi-model food selection."
    )


# ============================================================
# FINAL HYBRID DECISION
# ============================================================

def hybrid_decision(yolo, api1, api2):
    """
    Main V4 decision function.

    Input:
        yolo -> list of YOLO predictions
        api1 -> list of API 1 predictions
        api2 -> list of API 2 predictions

    Output:
        dictionary containing:
            food
            confidence
            decision
            supporting_models
            winner_margin
            decision_reason
            scores
    """

    # --------------------------------------------------------
    # STEP 1: Raw model fusion
    # --------------------------------------------------------

    original_scores = build_scores(
        yolo,
        api1,
        api2
    )

    if not original_scores:

        return {
            "food": "unknown",
            "confidence": 0.0,
            "decision": "NO PREDICTION",
            "supporting_models": 0,
            "winner_margin": 0.0,
            "decision_reason": "No valid model predictions.",
            "scores": {}
        }

    # --------------------------------------------------------
    # STEP 2: Side-dish handling
    # --------------------------------------------------------

    adjusted_scores = apply_side_dish_penalty(
        original_scores
    )

    # --------------------------------------------------------
    # STEP 3: Main-food prior
    # --------------------------------------------------------

    adjusted_scores = apply_main_food_prior(
        adjusted_scores
    )

    # --------------------------------------------------------
    # STEP 4: Specific conflict resolution
    # --------------------------------------------------------

    adjusted_scores = resolve_main_food_conflicts(
        adjusted_scores,
        original_scores
    )

    # --------------------------------------------------------
    # STEP 5: Remove negative values
    # --------------------------------------------------------

    for food in adjusted_scores:

        adjusted_scores[food] = max(
            0.0,
            adjusted_scores[food]
        )

    # --------------------------------------------------------
    # STEP 6: Sort candidates
    # --------------------------------------------------------

    ranked = sorted(
        adjusted_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    winner = ranked[0][0]
    winner_score = ranked[0][1]

    if len(ranked) > 1:
        runner_score = ranked[1][1]
    else:
        runner_score = 0.0

    winner_margin = winner_score - runner_score

    # --------------------------------------------------------
    # STEP 7: Supporting models
    # --------------------------------------------------------

    supporting_models = get_model_support(
        winner,
        yolo,
        api1,
        api2
    )

    # --------------------------------------------------------
    # STEP 8: Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        winner_score,
        runner_score,
        supporting_models,
        len(ranked)
    )

    # --------------------------------------------------------
    # STEP 9: Decision label
    # --------------------------------------------------------

    decision = get_decision(
        confidence,
        winner_margin
    )

    # --------------------------------------------------------
    # STEP 10: Decision reason
    # --------------------------------------------------------

    decision_reason = get_decision_reason(
        winner,
        original_scores,
        adjusted_scores
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "food": winner,
        "confidence": confidence,
        "decision": decision,
        "supporting_models": supporting_models,
        "winner_margin": round(winner_margin, 4),
        "decision_reason": decision_reason,

        # Return final adjusted scores.
        "scores": {
            food: round(score, 4)
            for food, score in ranked
        }
    }


# ============================================================
# END OF V4 ENGINE
# ============================================================