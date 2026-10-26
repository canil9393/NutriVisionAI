import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from hybrid_pipeline import run_hybrid
from food_instance_selector import select_food_instances, review_cross_label_overlaps
from food_segmenter import segment_food

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "segmentation_results" / "app_runs" / "latest"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_IMAGE_WIDTH = 650
OVERLAY_IMAGE_WIDTH = 650
INDIVIDUAL_MASK_WIDTH = 420

st.set_page_config(page_title="NutriVision AI", page_icon="🥗", layout="wide")
st.title("🥗 NutriVision AI")
st.caption("Hybrid food detection • SAM2 segmentation • Approximate nutrition")

# Clear results produced by older app versions after code updates.
APP_RESULT_VERSION = 4
if st.session_state.get("_nutrivision_result_version") != APP_RESULT_VERSION:
    st.session_state.pop("nutrivision_result", None)
    st.session_state["_nutrivision_result_version"] = APP_RESULT_VERSION

NUTRITION = {
    "alooparatha": (260, 5.5, 36, 10), "rasgulla": (186, 4, 38, 1),
    "biryani": (180, 6, 25, 6), "chickentikka": (165, 25, 4, 5),
    "palakpaneer": (160, 7, 6, 10), "poha": (160, 3, 27, 5),
    "khichdi": (130, 4, 22, 3), "omelette": (154, 11, 1, 12),
    "plainrice": (130, 2.7, 28, 0.3), "rice": (130, 2.7, 28, 0.3),
    "chapati": (297, 9.5, 50, 7), "chapathi": (297, 9.5, 50, 7),
    "bhatura": (300, 7, 43, 11), "dosa": (168, 4, 29, 4),
    "idli": (58, 2, 12, 0.4), "poori": (300, 6, 36, 15),
    "vada": (280, 7, 30, 14), "samosa": (300, 5, 32, 17),
    "coconutchutney": (250, 3, 8, 24), "greenchutney": (80, 3, 10, 3),
    "whitechutney": (150, 4, 12, 9), "redchutney": (120, 3, 15, 4),
    "sambar": (70, 3, 10, 2), "sambhar": (70, 3, 10, 2),
    "pickle": (100, 1, 8, 7), "curry": (140, 4, 10, 9),
    "chole": (165, 8, 27, 3), "kheer": (150, 4, 24, 4),
    "egg": (155, 13, 1.1, 11), "papadam": (350, 20, 60, 2),
    "papad": (350, 20, 60, 2), "curd": (61, 3.5, 4.7, 3.3),
    "yogurt": (61, 3.5, 4.7, 3.3),
}
DEFAULT_G = {
    "alooparatha": 150, "rasgulla": 50, "biryani": 250, "chickentikka": 150,
    "palakpaneer": 180, "poha": 200, "khichdi": 250, "omelette": 100,
    "plainrice": 180, "rice": 180, "chapati": 40, "chapathi": 40,
    "bhatura": 100, "dosa": 120, "idli": 40, "poori": 40, "vada": 60,
    "samosa": 100, "coconutchutney": 30, "greenchutney": 20,
    "whitechutney": 20, "redchutney": 20, "sambar": 150, "sambhar": 150,
    "pickle": 15, "curry": 150, "chole": 180, "kheer": 150, "egg": 50,
    "papadam": 15, "papad": 15, "curd": 100, "yogurt": 100,
}

def canon(name):
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {
        "plainrice": "rice", "whiterice": "rice",
        "chapathi": "chapati", "chappathi": "chapati",
        "sambhar": "sambar", "papadum": "papadam",
        "yoghurt": "yogurt",
        # Common spelling variants returned by food-recognition models.
        "biriyani": "biryani",
        "briyani": "biryani",
        "birani": "biryani",
    }
    return aliases.get(key, key)

def box_iou(box_a, box_b):
    """IoU for xyxy bounding boxes."""
    try:
        ax1, ay1, ax2, ay2 = map(float, box_a[:4])
        bx1, by1, bx2, by2 = map(float, box_b[:4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

def suppress_same_food_duplicates(detections, iou_threshold=0.50):
    """
    Cross-model non-maximum suppression for same-label predictions.
    Keeps the higher-confidence box when same-food boxes overlap strongly.
    Different labels are preserved for manual review.
    """
    ordered = sorted(
        detections,
        key=lambda d: float(d.get("confidence", 0.0)),
        reverse=True,
    )
    kept = []
    suppressed = []
    for det in ordered:
        label = canon(det.get("food", "unknown"))
        duplicate_of = None
        for prior in kept:
            if label == canon(prior.get("food", "unknown")):
                if box_iou(det.get("bbox", []), prior.get("bbox", [])) >= iou_threshold:
                    duplicate_of = prior
                    break
        if duplicate_of is None:
            kept.append(det)
        else:
            suppressed.append({
                "Food": det.get("food", "unknown"),
                "Suppressed model": det.get("source", "unknown"),
                "Suppressed confidence": f"{float(det.get('confidence', 0)):.1%}",
                "Kept model": duplicate_of.get("source", "unknown"),
                "Kept confidence": f"{float(duplicate_of.get('confidence', 0)):.1%}",
                "Box IoU": round(box_iou(det.get("bbox", []), duplicate_of.get("bbox", [])), 3),
            })
    return kept, suppressed

def mask_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0

def suppress_duplicate_masks(items, mask_iou_threshold=0.90):
    """
    After SAM2, suppress same-food instances whose masks almost completely
    overlap. Keep the higher-confidence detection and return an audit list.
    """
    ordered = sorted(
        items,
        key=lambda item: float(item.get("detection_confidence", 0.0)),
        reverse=True,
    )
    kept = []
    suppressed = []
    for item in ordered:
        label = canon(item.get("food", "unknown"))
        duplicate_of = None
        duplicate_iou = 0.0
        for prior in kept:
            if label == canon(prior.get("food", "unknown")):
                overlap = mask_iou(item.get("mask"), prior.get("mask"))
                if overlap >= mask_iou_threshold:
                    duplicate_of = prior
                    duplicate_iou = overlap
                    break
        if duplicate_of is None:
            kept.append(item)
        else:
            suppressed.append({
                "Food": item.get("food", "unknown"),
                "Suppressed model": item.get("source", "unknown"),
                "Suppressed confidence": f"{float(item.get('detection_confidence', 0)):.1%}",
                "Kept model": duplicate_of.get("source", "unknown"),
                "Kept confidence": f"{float(duplicate_of.get('detection_confidence', 0)):.1%}",
                "Mask IoU": round(duplicate_iou, 3),
            })
    return kept, suppressed

def rebuild_overlay(image_path, items, overlay_path):
    """Rebuild overlay from only retained masks so removed duplicates disappear."""
    base = Image.open(image_path).convert("RGB")
    arr = np.asarray(base).copy()
    tint = np.zeros_like(arr)
    tint[:, :, 1] = 255
    combined = np.zeros(arr.shape[:2], dtype=bool)
    for item in items:
        mask = np.asarray(item.get("mask"), dtype=bool)
        if mask.shape == combined.shape:
            combined |= mask
    if combined.any():
        arr[combined] = (0.45 * arr[combined] + 0.55 * tint[combined]).astype(np.uint8)
    Image.fromarray(arr).save(overlay_path)
    return str(overlay_path)

def analyze(image_path):
    hybrid = run_hybrid(str(image_path))
    candidates = []
    for source in ("yolo", "api1", "api2"):
        for raw in hybrid.get(source, []):
            item = dict(raw)
            item["source"] = source
            candidates.append(item)

    # Primary-model policy: if YOLO11 has a prediction at or above 50%,
    # choose only its highest-confidence prediction for the main result.
    # Roboflow is a fallback only when YOLO11 has no qualifying prediction.
    min_primary_confidence = 0.50
    yolo_candidates = [
        item for item in candidates
        if item.get("source") == "yolo"
        and float(item.get("confidence", 0.0)) >= min_primary_confidence
    ]
    if yolo_candidates:
        chosen = max(yolo_candidates, key=lambda item: float(item.get("confidence", 0.0)))
        selected = [chosen]
        selection_mode = "YOLO11 primary"
    else:
        api_candidates = [
            item for item in candidates
            if item.get("source") in ("api1", "api2")
            and float(item.get("confidence", 0.0)) >= min_primary_confidence
        ]
        selected = [max(api_candidates, key=lambda item: float(item.get("confidence", 0.0)))] if api_candidates else []
        selection_mode = "Roboflow fallback (no YOLO11 prediction ≥ 50%)"

    # Record non-selected predictions for transparency, but do not segment or
    # count them in nutrition. This avoids treating other models' labels as
    # extra foods in a single-food image.
    selected_identity = id(selected[0]) if selected else None
    selected_signature = (
        selected[0].get("source"),
        str(selected[0].get("food", "")).lower(),
        tuple(round(float(v), 2) for v in selected[0].get("bbox", [])),
    ) if selected else None
    other_predictions = []
    for item in candidates:
        signature = (
            item.get("source"),
            str(item.get("food", "")).lower(),
            tuple(round(float(v), 2) for v in item.get("bbox", [])),
        )
        if signature != selected_signature:
            other_predictions.append({
                "Food": item.get("food", "unknown"),
                "Model": item.get("source", "unknown"),
                "Confidence": f"{float(item.get('confidence', 0.0)):.1%}",
            })

    conflicts = review_cross_label_overlaps(selected, iou_threshold=0.85) if selected else []
    segmented = segment_food(str(image_path), selected, output_dir=str(OUTPUT_DIR)) if selected else {
        "success": False, "message": "No prediction reached the 50% confidence threshold.",
        "items": [], "overlay_path": ""
    }
    suppressed_masks = []
    if segmented.get("items"):
        retained_items, suppressed_masks = suppress_duplicate_masks(
            segmented["items"], mask_iou_threshold=0.90
        )
        segmented["items"] = retained_items
        segmented["success"] = bool(retained_items)
        if retained_items:
            segmented["overlay_path"] = rebuild_overlay(
                image_path, retained_items, segmented["overlay_path"]
            )
        elif not segmented.get("message"):
            segmented["message"] = "No non-duplicate masks remained after filtering."
    return candidates, selected, conflicts, segmented, suppressed_masks, other_predictions, selection_mode

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        st.stop()

    st.image(image, caption="Uploaded image", width=UPLOAD_IMAGE_WIDTH)

    if st.button("Analyze food", type="primary"):
        suffix = Path(uploaded.name).suffix.lower() or ".jpg"
        image_path = UPLOAD_DIR / f"nutrivision_input{suffix}"
        image.save(image_path)
        with st.spinner("Running YOLO11, Roboflow APIs and SAM2. First run may take longer..."):
            try:
                candidates, selected, conflicts, segmented, suppressed_masks, other_predictions, selection_mode = analyze(image_path)
                st.session_state["_nutrivision_result_version"] = APP_RESULT_VERSION
                st.session_state["nutrivision_result"] = {
                    "candidates": candidates,
                    "selected": selected,
                    "conflicts": conflicts,
                    "segmented": segmented,
                    "suppressed_duplicates": other_predictions,
                    "suppressed_mask_duplicates": suppressed_masks,
                    "selection_mode": selection_mode,
                }
            except Exception as exc:
                st.exception(exc)

if "nutrivision_result" in st.session_state:
    result = st.session_state["nutrivision_result"]
    selected, segmented = result["selected"], result["segmented"]

    if result.get("suppressed_mask_duplicates"):
        with st.expander(
            f"Same-food duplicate masks removed ({len(result['suppressed_mask_duplicates'])})"
        ):
            st.caption(
                "Same-label masks with IoU ≥ 0.90 are treated as likely duplicate "
                "predictions. Separate servings can occasionally overlap, so review this list."
            )
            st.dataframe(result["suppressed_mask_duplicates"], width="stretch")

    st.subheader("1. Food detection")
    st.write(
        f"Raw detections: **{len(result['candidates'])}** · "
        f"Selected after duplicate filtering: **{len(selected)}**"
    )

    if result.get("suppressed_duplicates"):
        with st.expander(
            f"Same-food duplicate predictions removed ({len(result['suppressed_duplicates'])})"
        ):
            st.caption(
                "Predictions with the same normalized food label and bounding-box IoU "
                "of at least 0.50 are treated as likely duplicates. Check these if your "
                "image contains separate servings close together."
            )
            st.dataframe(result["suppressed_duplicates"], width="stretch")

    if selected:
        st.dataframe(
            [{
                "Food": d.get("food", "unknown"),
                "Model": d.get("source", "unknown"),
                "Confidence": f"{float(d.get('confidence', 0)):.1%}",
                "Box": ", ".join(str(round(v)) for v in d["bbox"]),
            } for d in selected],
            width="stretch",
        )
    else:
        st.warning("No detections passed the confidence filter. Try a clear, well-lit image.")

    if result["conflicts"]:
        st.warning("Different-label box overlaps need review; they have not been automatically deleted.")
        st.dataframe(result["conflicts"], width="stretch")

    # Final safety pass at render time: also deduplicate masks in case a cached
    # result or older pipeline path still contains duplicate same-food masks.
    if segmented.get("items"):
        display_items, display_mask_duplicates = suppress_duplicate_masks(
            segmented["items"], mask_iou_threshold=0.90
        )
        if len(display_items) != len(segmented["items"]):
            segmented["items"] = display_items
            segmented["success"] = bool(display_items)
            if display_items:
                segmented["overlay_path"] = rebuild_overlay(
                    UPLOAD_DIR / f"nutrivision_input{Path(uploaded.name).suffix.lower() or '.jpg'}",
                    display_items,
                    segmented["overlay_path"],
                )
        if display_mask_duplicates:
            with st.expander(
                f"Duplicate same-food masks excluded ({len(display_mask_duplicates)})"
            ):
                st.caption(
                    "Same-label masks with IoU ≥ 0.90 were treated as duplicate predictions. "
                    "Review this list if separate servings overlap."
                )
                st.dataframe(display_mask_duplicates, width="stretch")

    st.subheader("2. SAM2 segmentation")
    if segmented.get("success") and Path(segmented["overlay_path"]).exists():
        st.image(segmented["overlay_path"], caption="Combined mask overlay", width=OVERLAY_IMAGE_WIDTH)
        with st.expander("Show individual masks"):
            for item in segmented["items"]:
                st.write(
                    f"**{item['food']}** · {item['source']} · "
                    f"mask coverage {item['mask_coverage_ratio']:.1%}"
                )
                if Path(item["mask_path"]).exists():
                    st.image(item["mask_path"], caption=f"Mask: {item['food']}", width=INDIVIDUAL_MASK_WIDTH)
    else:
        st.warning(segmented.get("message", "No valid masks were generated."))

    st.subheader("3. Portion and nutrition estimate")
    st.caption(
        "Nutrition values are generic reference averages per 100 g. "
        "Portion weights are editable estimates, not measured weights."
    )
    items = segmented.get("items", [])
    if not items:
        st.info("Nutrition totals are unavailable until at least one valid mask is generated.")
    else:
        conflict_indices = set()
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                overlap = mask_iou(items[a]["mask"], items[b]["mask"])
                if overlap >= 0.90 and canon(items[a]["food"]) != canon(items[b]["food"]):
                    conflict_indices.update((a, b))
                    st.warning(
                        f"Mask conflict: {items[a]['food']} vs {items[b]['food']} "
                        f"(IoU {overlap:.3f}). Inspect the masks before counting both."
                    )

        portion_data = []
        for i, item in enumerate(items):
            label = canon(item.get("food", "unknown"))
            default_g = int(DEFAULT_G.get(label, 100))
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.write(f"**{item['food']}**")
                    st.caption(
                        f"{item['source']} · detection confidence "
                        f"{item['detection_confidence']:.1%}"
                    )
                with c2:
                    grams = st.slider(
                        f"Estimated portion (g) — {item['food']} #{i+1}",
                        min_value=10, max_value=500,
                        value=min(500, max(10, default_g)), step=5,
                        key=f"portion_{i}_{label}",
                    )
                values = NUTRITION.get(label)
                with c3:
                    if values:
                        st.metric("Estimated kcal", f"{values[0] * grams / 100:.0f}")
                    else:
                        st.warning("No nutrition record")
            portion_data.append((i, item, label, grams, values))

        included = []
        for i, item, label, grams, values in portion_data:
            checkbox_key = f"include_{i}_{label}"
            if checkbox_key not in st.session_state:
                st.session_state[checkbox_key] = i not in conflict_indices
            include = st.checkbox(
                f"Include {item['food']} in nutrition totals",
                key=checkbox_key,
            )
            if include and values:
                included.append((grams, values))

        total_kcal = total_protein = total_carbs = total_fat = 0.0
        for grams, values in included:
            kcal, protein, carbs, fat = values
            factor = grams / 100.0
            total_kcal += kcal * factor
            total_protein += protein * factor
            total_carbs += carbs * factor
            total_fat += fat * factor

        st.subheader("Estimated nutrition total")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", f"{total_kcal:.0f} kcal")
        c2.metric("Protein", f"{total_protein:.1f} g")
        c3.metric("Carbohydrates", f"{total_carbs:.1f} g")
        c4.metric("Fat", f"{total_fat:.1f} g")
        st.caption(
            f"{len(included)} item(s) included. Values vary by recipe, ingredients, and preparation."
        )

st.divider()
st.caption("NutriVision AI · Hybrid recognition + SAM2 segmentation + editable nutrition estimates")