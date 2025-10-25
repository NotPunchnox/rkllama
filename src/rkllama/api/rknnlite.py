# img_encoder.py
import numpy as np
import logging
import cv2
import base64
import requests
from rknnlite.api.rknn_lite import RKNNLite    
import os

logger = logging.getLogger("rkllama.rknnlite")


# Run the visionencder to get the image embedding
def run_vision_encoder(model_encoder_path, images_source, image_width, image_height):
    """
    Run the vision encoder to get the image embedding
    Args:
        model_encoder_path (str): Path of the encoder
        image_source (str): Image source (base64, PATH or URL)
        image_width (int): Width of the image
        image_height (int): Height of the image
    Returns:
        np.ndarray: Image embedding
    """
    
    # Prepare the image
    prepared_images = [prepare_image(image_source, image_width, image_height) for image_source in images_source]

    # Init encoder
    vision_encoder = RKNNLite(verbose=False)
    vision_encoder.load_rknn(model_encoder_path)
    vision_encoder.init_runtime()

    # Inference
    image_embeddings = [vision_encoder.inference(inputs=[img.astype(np.float32)], data_type="float32", data_format="nhwc")[0] for img in prepared_images]
    logger.debug(f"Image embeddings shapes: {[emb.shape for emb in image_embeddings]}")

    # Concatenate along the first axis (rows)
    np_float32_image_embeddings = [emb.astype(np.float32) for emb in image_embeddings]
    concatenated_image_embeddings = np.concatenate(np_float32_image_embeddings, axis=0)
    logger.debug(f"Concatenated image embeddings shape: {concatenated_image_embeddings.shape}")

    # Release RKNNLite resources
    vision_encoder.release()

    # Return the encoded image to the main process
    return concatenated_image_embeddings


def load_image(source: str):
    """
    Load an image from:
      - a local path
      - a URL
      - a Base64 string
    Returns:
      - image as numpy array (BGR) or None if fails
    """
    img = None
    
    # Case 1: local file
    if os.path.exists(source):
        img = cv2.imread(source)
    
    # Case 2: URL
    elif source.startswith("http://") or source.startswith("https://"):
        try:
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Error loading from URL:", e)
    
    # Case 3: Base64
    else:
        try:
            # Remove "data:image/..;base64," if present
            if "," in source:
                source = source.split(",")[1]
            img_data = base64.b64decode(source)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Error loading from Base64:", e)
    
    return img


def prepare_image(image_source, image_width, image_height) -> np.ndarray:
    """ Load and preprocess an image for model input.
        Args:
            image_source: Path, URL, or Base64 string of the image.
            image_width: Target width for resizing.
            image_height: Target height for resizing.
        Returns:
            Preprocessed image as a numpy array (HWC, uint8).
    """
    # Read image
    img = load_image(image_source)  # BGR
    if img is None:
        raise FileNotFoundError(image_source)
    
    # Preprocess Image 
    resized = cv2.resize(img, (image_width, image_height))
    resized = resized.astype(np.float32)
    resized = resized[np.newaxis, :, :, :]
    
    # Return the image
    return resized

