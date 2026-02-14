

import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from config import IMAGE_SIZE


def preprocess(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    img = np.array(image).astype(np.float32)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)
    return img
