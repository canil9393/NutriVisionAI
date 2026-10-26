import os

import numpy as np
import streamlit as st
from PIL import Image

from utils.image_preprocessing import preprocess_image
from utils.food_detector import FoodDetector


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🍽️",
    layout="wide"
)


# --------------------------------------------------
# Initialize YOLO Detector
# --------------------------------------------------

detector = FoodDetector()


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("🍽️ NutriVision AI")

st.write(
    "AI-Based Food Nutrition Estimation from a Food Image"
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("NutriVision AI")

st.sidebar.success("Module 3: Food Detection")

st.sidebar.info(
    "Upload a food image to detect objects using YOLO."
)


# --------------------------------------------------
# Upload Folder
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload Food Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Process Uploaded Image
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Convert image to RGB
    image = image.convert("RGB")


    # --------------------------------------------------
    # Save Uploaded Image
    # --------------------------------------------------

    file_path = os.path.join(
        UPLOAD_DIR,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Image uploaded successfully!")


    # --------------------------------------------------
    # Module 2 - Image Preprocessing
    # --------------------------------------------------

    processed_image, normalized_image = preprocess_image(
        image
    )


    st.subheader("Image Preprocessing")

    col1, col2 = st.columns(2)


    with col1:

        st.write("Original Image")

        st.image(
            image,
            use_container_width=True
        )


    with col2:

        st.write("Processed Image")

        st.image(
            processed_image,
            channels="BGR",
            use_container_width=True
        )


    # --------------------------------------------------
    # Image Information
    # --------------------------------------------------

    st.subheader("Image Information")

    original_width, original_height = image.size

    processed_height, processed_width = (
        processed_image.shape[:2]
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Original Width",
            f"{original_width}px"
        )


    with col2:

        st.metric(
            "Original Height",
            f"{original_height}px"
        )


    with col3:

        st.metric(
            "Processed Size",
            f"{processed_width} × {processed_height}"
        )


    # --------------------------------------------------
    # Module 3 - YOLO Detection
    # --------------------------------------------------

    st.divider()

    st.subheader("Food Detection")


    # Convert PIL image to NumPy array
    image_array = np.array(image)


    # Run YOLO
    results = detector.detect(
        image_array
    )


    # --------------------------------------------------
    # Display Detection Image
    # --------------------------------------------------

    for result in results:

        detected_image = result.plot()

        st.image(
            detected_image,
            caption="YOLO Detection",
            channels="BGR",
            use_container_width=True
        )


    # --------------------------------------------------
    # Extract Detected Objects
    # --------------------------------------------------

    detected_objects = []


    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = result.names[
                class_id
            ]


            detected_objects.append(
                {
                    "name": class_name,
                    "confidence": confidence
                }
            )


    # --------------------------------------------------
    # Display Detected Objects
    # --------------------------------------------------

    st.subheader(
        "Detected Objects"
    )


    if detected_objects:

        for obj in detected_objects:

            st.write(
                f"**{obj['name']}** "
                f"— {obj['confidence']:.2%}"
            )

    else:

        st.info(
            "No objects detected."
        )


    # --------------------------------------------------
    # Nutrition Placeholder
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "Nutrition Analysis"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Calories",
            "-- kcal"
        )


    with col2:

        st.metric(
            "Protein",
            "-- g"
        )


    with col3:

        st.metric(
            "Carbohydrates",
            "-- g"
        )


    with col4:

        st.metric(
            "Fat",
            "-- g"
        )


else:

    st.info(
        "Please upload a food image "
        "to start the analysis."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "NutriVision AI | Module 3 - Food Detection"
)