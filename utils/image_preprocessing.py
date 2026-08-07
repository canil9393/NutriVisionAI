import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess an uploaded food image.

    Steps:
    1. Convert image to RGB
    2. Resize image to 640x640
    3. Normalize pixel values
    """

    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize image
    image = cv2.resize(image, (640, 640))

    # Normalize pixel values
    normalized_image = image / 255.0

    return image, normalized_image