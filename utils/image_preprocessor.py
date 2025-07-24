from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocessing gambar untuk input ke CNN (VGG16):
    - Resize gambar
    - Konversi ke RGB
    - Normalisasi (preprocess_input dari VGG16)
    - Tambahkan batch dimension

    Args:
        image (PIL.Image): Gambar input dari pengguna
        target_size (tuple): Ukuran input model, default (224, 224)

    Returns:
        np.ndarray: Array bentuk (1, 224, 224, 3) siap untuk prediksi
    """
    # Resize dan pastikan RGB
    image = image.resize(target_size)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Konversi ke array dan preprocessing
    img_array = np.array(image).astype("float32")
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    # Tambah dimensi batch
    return np.expand_dims(img_array, axis=0)