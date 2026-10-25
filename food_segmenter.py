
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

SAM_MODEL_NAME = "sam2.1_b.pt"
_model = None


def load_segmenter():
    """Load SAM2 once and reuse the model."""
    global _model

    if _model is None:
        _model = SAM(SAM_MODEL_NAME)

    return _model


def clamp_bbox(bbox, image_width, image_height):
    """Clamp a bounding box to the image boundaries."""
    x1, y1, x2, y2 = map(float, bbox)

    x1 = max(0, min(int(round(x1)), image_width - 1))
    y1 = max(0, min(int(round(y1)), image_height - 1))
    x2 = max(0, min(int(round(x2)), image_width))
    y2 = max(0, min(int(round(y2)), image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def get_adaptive_prompt_points(bbox, image_width, image_height):
    """
    Return one positive point near the center of a bounding box.
    Coordinates are in the original image coordinate system.
    """
    box = clamp_bbox(bbox, image_width, image_height)

    if box is None:
        return []

    x1, y1, x2, y2 = box

    point_x = int(round((x1 + x2) / 2))
    point_y = int(round((y1 + y2) / 2))

    point_x = max(0, min(point_x, image_width - 1))
    point_y = max(0, min(point_y, image_height - 1))

    return [[point_x, point_y]]


def get_food_prompts(detection, bbox, image_width, image_height):
    """
    Generate prompts for the detected food.

    Positive points are calculated from the detection box.
    For flatbreads, a negative point is placed toward an edge of
    the box to discourage SAM2 from selecting a neighboring item.

    Note: box-derived points cannot guarantee that a point falls
    on the correct food when the detection box contains multiple items.
    """
    food_name = str(detection.get("food", "unknown")).strip().lower()

    positive_points = get_adaptive_prompt_points(
        bbox, image_width, image_height
    )

    if not positive_points:
        return None, None

    if food_name in ("chapati", "bhatura"):
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        # Choose an edge-relative negative point.
        # It is only a heuristic; inspect results on each image.
        negative_x = int(round(x1 + 0.12 * box_width))
        negative_y = int(round(y1 + 0.80 * box_height))

        negative_x = max(0, min(negative_x, image_width - 1))
        negative_y = max(0, min(negative_y, image_height - 1))

        positive_x, positive_y = positive_points[0]

        points = [[
            [positive_x, positive_y],
            [negative_x, negative_y],
        ]]
        labels = [[1, 0]]

        return points, labels

    return [positive_points], [[1]]


def resize_mask(mask, image_width, image_height):
    """Resize a SAM mask to original image dimensions."""
    return cv2.resize(
        mask.astype("uint8"),
        (image_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def select_best_mask(
    masks,
    bbox,
    positive_point,
    image_width,
    image_height,
):
    """
    Select the mask with the best combination of:
    - fraction of mask area inside the detection box;
    - coverage of the detection box;
    - inclusion of the positive prompt.

    This does not guarantee semantic correctness, but is more robust
    than blindly selecting the first or highest-scoring mask.
    """
    x1, y1, x2, y2 = bbox
    positive_x, positive_y = positive_point

    best_mask = None
    best_score = -1.0

    for mask in masks:
        resized = resize_mask(mask, image_width, image_height)

        mask_area = int(np.count_nonzero(resized))
        if mask_area == 0:
            continue

        inside_box = resized[y1:y2, x1:x2]
        inside_area = int(np.count_nonzero(inside_box))

        if inside_area == 0:
            continue

        box_area = max(1, (x2 - x1) * (y2 - y1))
        box_coverage = inside_area / box_area
        mask_precision = inside_area / mask_area

        point_inside = bool(resized[positive_y, positive_x])
        if not point_inside:
            continue

        # Favor masks mostly contained by the box while retaining
        # reasonable coverage of the detected food.
        score = (
            0.60 * mask_precision
            + 0.40 * min(box_coverage, 1.0)
        )

        if score > best_score:
            best_score = score
            best_mask = resized

    return best_mask


def segment_food(image_path, detections, output_dir="segmentation_results"):
    image_path = str(image_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(image_path)

    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = original.shape[:2]
    model = load_segmenter()

    items = []
    combined_mask = np.zeros((height, width), dtype=bool)

    for index, detection in enumerate(detections, start=1):
        raw_bbox = detection.get("bbox")

        if not raw_bbox or len(raw_bbox) != 4:
            print("Skipping detection with invalid bbox:", detection)
            continue

        bbox = clamp_bbox(raw_bbox, width, height)

        if bbox is None:
            print("Skipping empty bbox:", detection.get("food", "unknown"))
            continue

        x1, y1, x2, y2 = bbox
        food_name = str(detection.get("food", "unknown")).strip().lower()

        points, labels = get_food_prompts(
            detection, bbox, width, height
        )

        if points is None:
            print("Skipping prompt generation for:", food_name)
            continue

        predict_args = {
            "source": image_path,
            "bboxes": [[x1, y1, x2, y2]],
            "points": points,
            "labels": labels,
            "imgsz": 1024,
            "retina_masks": True,
            "conf": 0.10,
            "verbose": False,
        }

        print(
            f"Prompt | food={food_name} | "
            f"bbox={bbox} | points={points} | labels={labels}"
        )

        results = model.predict(**predict_args)

        if not results or results[0].masks is None:
            print(f"No mask produced for {food_name}")
            continue

        result = results[0]
        masks = result.masks.data.cpu().numpy()

        print(f"{food_name}: {len(masks)} candidate mask(s)")

        positive_point = points[0][0]

        selected = select_best_mask(
            masks=masks,
            bbox=bbox,
            positive_point=positive_point,
            image_width=width,
            image_height=height,
        )

        if selected is None:
            print(f"No suitable mask selected for {food_name}")
            continue

        # Constrain the selected mask to its detection box.
        clipped_mask = np.zeros((height, width), dtype=bool)
        clipped_mask[y1:y2, x1:x2] = selected[y1:y2, x1:x2]

        pixel_area = int(np.count_nonzero(clipped_mask))
        bbox_area = int((x2 - x1) * (y2 - y1))

        coverage_ratio = (
            pixel_area / bbox_area if bbox_area > 0 else 0.0
        )

        print(
            f"{detection.get('food', 'unknown')} | "
            f"Box area: {bbox_area} px | "
            f"Mask area: {pixel_area} px | "
            f"Coverage: {coverage_ratio:.3f}"
        )

        if pixel_area == 0:
            print(f"Empty mask for {food_name}")
            continue

        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in food_name
        )

        mask_path = output_path / f"{index}_{safe_name}_mask.png"

        cv2.imwrite(
            str(mask_path),
            clipped_mask.astype("uint8") * 255,
        )

        combined_mask |= clipped_mask

        items.append({
            "food": food_name,
            "source": detection.get("source", "unknown"),
            "detection_confidence": float(
                detection.get("confidence", 0)
            ),
            "bbox_area_pixels": bbox_area,
            "mask_area_pixels": pixel_area,
            "mask_coverage_ratio": round(coverage_ratio, 4),
            "mask": clipped_mask,
            "mask_path": str(mask_path),
        })

    # Save a green overlay of all selected masks.
    overlay = original.copy()
    green = np.zeros_like(original)
    green[:] = (0, 255, 0)

    overlay[combined_mask] = (
        0.5 * original[combined_mask]
        + 0.5 * green[combined_mask]
    ).astype("uint8")

    overlay_path = output_path / (
        Path(image_path).stem + "_segmented.jpg"
    )

    cv2.imwrite(str(overlay_path), overlay)

    return {
        "success": bool(items),
        "message": (
            "Segmentation completed."
            if items
            else "No valid segmentation masks were generated."
        ),
        "items": items,
        "overlay_path": str(overlay_path),
    }
