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

st.set_page_config(page_title="NutriVision AI", page_icon="🥗", layout="wide")
st.title("🥗 NutriVision AI")
st.caption("Hybrid food detection • SAM2 segmentation • Approximate nutrition")

# Generic reference averages per 100 g; recipes vary substantially.
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
    "sambar": (70, 3, 10, 2), "pickle": (100, 1, 8, 7),
    "curry": (140, 4, 10, 9), "chole": (165, 8, 27, 3),
    "kheer": (150, 4, 24, 4), "egg": (155, 13, 1.1, 11),
    "papadam": (350, 22, 58, 2), "papad": (350, 22, 58, 2),
    "curd": (61, 3.5, 4.7, 3.3), "yogurt": (61, 3.5, 4.7, 3.3),
}
DEFAULT_G = {
    "alooparatha": 150, "rasgulla": 50, "biryani": 250, "chickentikka": 150,
    "palakpaneer": 180, "poha": 200, "khichdi": 250, "omelette": 100,
    "plainrice": 180, "rice": 180, "chapati": 40, "chapathi": 40,
    "bhatura": 100, "dosa": 120, "idli": 40, "poori": 40, "vada": 60,
    "samosa": 100, "coconutchutney": 30, "greenchutney": 20,
    "whitechutney": 20, "redchutney": 20, "sambar": 150, "pickle": 15,
    "curry": 150, "chole": 180, "kheer": 150, "egg": 50,
    "papadam": 15, "papad": 15, "curd": 100, "yogurt": 100,
}

def canon(name):
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return {"plainrice": "rice", "whiterice": "rice",
            "chapathi": "chapati", "chappathi": "chapati"}.get(key, key)

def mask_iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0

def analyze(image_path):
    hybrid = run_hybrid(str(image_path))
    candidates = []
    for source in ("yolo", "api1", "api2"):
        for raw in hybrid.get(source, []):
            item = dict(raw)
            item["source"] = source
            candidates.append(item)
    selected = select_food_instances(candidates, min_confidence=0.50, duplicate_iou=0.85)
    conflicts = review_cross_label_overlaps(selected, iou_threshold=0.85)
    segmented = segment_food(str(image_path), selected, output_dir=str(OUTPUT_DIR))
    return candidates, selected, conflicts, segmented

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        st.stop()
    st.image(image, caption="Uploaded image", width=650)

    if st.button("Analyze food", type="primary"):
        suffix = Path(uploaded.name).suffix.lower() or ".jpg"
        image_path = UPLOAD_DIR / f"nutrivision_input{suffix}"
        image.save(image_path)
        with st.spinner("Running YOLO11, Roboflow APIs and SAM2. First run may take longer..."):
            try:
                candidates, selected, conflicts, segmented = analyze(image_path)
                st.session_state["nutrivision_result"] = {
                    "candidates": candidates, "selected": selected,
                    "conflicts": conflicts, "segmented": segmented,
                }
            except Exception as exc:
                st.exception(exc)

if "nutrivision_result" in st.session_state:
    result = st.session_state["nutrivision_result"]
    selected, segmented = result["selected"], result["segmented"]

    st.subheader("1. Food detection")
    st.write(f"Raw detections: **{len(result['candidates'])}** · Selected candidates: **{len(selected)}**")
    if selected:
        st.dataframe([{
            "Food": d.get("food", "unknown"),
            "Model": d.get("source", "unknown"),
            "Confidence": f"{float(d.get('confidence', 0)):.1%}",
            "Box": ", ".join(str(round(v)) for v in d["bbox"]),
        } for d in selected], use_container_width=True)
    else:
        st.warning("No detections passed the confidence filter. Try a clear, well-lit image.")

    if result["conflicts"]:
        st.warning("Different-label box overlaps need review; they have not been automatically deleted.")
        st.dataframe(result["conflicts"], use_container_width=True)

    st.subheader("2. SAM2 segmentation")
    if segmented.get("success") and Path(segmented["overlay_path"]).exists():
        st.image(segmented["overlay_path"], caption="Combined mask overlay (preview)", width=650)
        st.caption("Previews are limited to 650 px wide. Green areas are SAM2 masks; overlapping masks can indicate segmentation errors.")
        with st.expander("Show individual masks"):
            for item in segmented["items"]:
                st.write(f"**{item['food']}** · {item['source']} · mask coverage {item['mask_coverage_ratio']:.1%}")
                if Path(item["mask_path"]).exists():
                    st.image(item["mask_path"], caption=f"Mask: {item['food']}", width=420)
    else:
        st.warning(segmented.get("message", "No valid masks were generated."))

    st.subheader("3. Portion and nutrition estimate")
    st.caption("Nutrition values are generic reference averages per 100 g. Portion weights are editable estimates, not measured weights.")
    items = segmented.get("items", [])
    if not items:
        st.info("Nutrition totals are unavailable until at least one valid mask is generated.")
    else:
        portion_data = []
        for i, item in enumerate(items):
            label = canon(item.get("food", "unknown"))
            default_g = int(DEFAULT_G.get(label, 100))
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.write(f"**{item['food']}**")
                    st.caption(f"{item['source']} · detection confidence {item['detection_confidence']:.1%}")
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

        included = []
        for i, item, label, grams, values in portion_data:
            include = st.checkbox(
                f"Include {item['food']} in nutrition totals",
                value=(i not in conflict_indices),
                key=f"include_{i}_{label}",
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
        st.caption(f"{len(included)} item(s) included. Values vary by recipe, ingredients, and preparation.")

st.divider()
st.caption("NutriVision AI · Hybrid recognition + SAM2 segmentation + editable nutrition estimates")
