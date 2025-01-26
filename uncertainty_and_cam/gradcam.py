import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Step 3: Grad-CAM Visualization
# def grad_cam(model, image, class_idx, last_conv_layer_name):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], 
#         [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(image)
#         loss = predictions[:, class_idx]
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
#     return heatmap

# def overlay_heatmap(image, heatmap, alpha=0.4, cmap='viridis'):
#     heatmap = plt.cm.get_cmap(cmap)(heatmap)[..., :3]
#     heatmap = tf.image.resize(heatmap, (image.shape[1], image.shape[2]))
#     overlay = heatmap * alpha + image
#     return np.clip(overlay, 0, 1)

# Grad-CAM Function
def grad_cam(model, image, class_idx, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU operation
    heatmap = heatmap / tf.reduce_max(heatmap)  # Normalize
    return heatmap.numpy()

# Overlay Heatmap Function
# def overlay_heatmap(image, heatmap, alpha=0.4, cmap='viridis'):
#     # Resize the heatmap to match the input image size
#     heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image.shape[1], image.shape[2]))
#     heatmap = tf.squeeze(heatmap).numpy()

#     # Normalize heatmap (0 to 1)
#     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

#     # Convert heatmap to RGB using colormap
#     heatmap = plt.cm.get_cmap(cmap)(heatmap)[..., :3]

#     # Overlay heatmap on the original image
#     if image.dtype != np.float32:
#         image = image.astype('float32') / 255.0  # Normalize the image (0 to 1)

#     overlay = (1 - alpha) * image + alpha * heatmap
#     overlay = np.clip(overlay, 0, 1)  # Ensure valid range [0, 1]
#     return overlay

def overlay_heatmap(image, heatmap, alpha=0.4, cmap='viridis'):
    # Resize the heatmap to match the input image size
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image.shape[0], image.shape[1]))
    heatmap = tf.squeeze(heatmap).numpy()  # Squeeze the last dimension to get (64, 64)

    # Normalize heatmap (0 to 1)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

    # Convert heatmap to RGB using colormap
    heatmap_rgb = plt.cm.get_cmap(cmap)(heatmap)[..., :3]  # Shape: (64, 64, 3)

    # Overlay heatmap on the original image
    if image.dtype != np.float32:
        image = image.astype('float32') / 255.0  # Normalize the image (0 to 1)

    overlay = (1 - alpha) * image + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)  # Ensure valid range [0, 1]
    return overlay

# Visualize Grad-CAM for all transformed test images
def plot_all_gradcams(model, images, mean_preds, last_conv_layer_name, output_dir, labels=None):
    """
    Plots and saves Grad-CAM overlays for all test images.

    Args:
        model: Trained model.
        images: Transformed test images (4D array).
        mean_preds: Predicted transformation class probabilities.
        last_conv_layer_name: Name of the last convolutional layer.
        output_dir: Directory to save the plots.
        labels: True class labels for the original images.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Predicted class index
        class_idx = np.argmax(mean_preds[i])

        # Compute Grad-CAM
        heatmap = grad_cam(model, image, class_idx, last_conv_layer_name)
        overlay = overlay_heatmap(image[0], heatmap)

        # Plot and save
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        title = f"Grad-CAM: Image {i+1} (Pred: {class_idx})"
        if labels is not None and i < len(labels):  # Ensure no out-of-bounds access
            try:
                title += f", True: {labels[i]}"
            except Exception as e:
                print(f"Error: {e}")
        plt.title(title)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f'gradcam_image_{i+1}.png'))
        plt.close()
