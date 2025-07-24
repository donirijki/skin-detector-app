import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    """
    Membuat heatmap Grad-CAM dari prediksi model.
    
    Args:
        img_array: Gambar input dalam bentuk array dengan shape (1, 224, 224, 3).
        model: Model CNN terlatih.
        last_conv_layer_name: Nama layer konvolusi terakhir.
        pred_index: Index kelas yang ingin divisualisasikan. Jika None, ambil prediksi tertinggi.

    Returns:
        heatmap: Array 2D normalisasi (0-1) untuk divisualisasikan.
    """
    # Model Grad-CAM
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Hitung gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradien dan pooling
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Buat heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())

    return heatmap.numpy()

def apply_heatmap_on_image(image: Image.Image, heatmap, alpha=0.4):
    """
    Overlay heatmap ke gambar asli.
    
    Args:
        image: Gambar PIL.
        heatmap: Heatmap 2D.
        alpha: Transparansi heatmap.

    Returns:
        Image PIL dengan overlay heatmap.
    """
    # Resize image dan heatmap
    image = image.resize((224, 224)).convert("RGB")
    img = np.array(image)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed_img)