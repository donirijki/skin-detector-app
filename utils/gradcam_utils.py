import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    """
    Membuat heatmap Grad-CAM dari prediksi model.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Hitung gradien
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # shape (H, W, Channels)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalisasi agar antara 0–1
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val

    return heatmap.numpy()

def apply_heatmap_on_image(image: Image.Image, heatmap, alpha=0.4):
    """
    Overlay heatmap ke gambar asli.
    """
    image = image.resize((224, 224)).convert("RGB")
    img = np.array(image)

    # Pastikan heatmap 2D
    if len(heatmap.shape) != 2:
        heatmap = np.squeeze(heatmap)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    return Image.fromarray(superimposed_img)