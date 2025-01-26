
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
import cv2  # For drawing bounding boxes

from model_infos import generate_layer_summary, model_summary

def add_colorbar(im, width=None, pad=None, **kwargs):

    l, b, w, h = im.axes.get_position().bounds       # get boundaries
    width = width or 0.1 * w                         # get width of the colorbar
    pad = pad or width                               # get pad between im and cbar
    fig = im.axes.figure                             # get figure of image
    cax = fig.add_axes([l + w + pad, b, width, h])   # define cbar Axes
    return fig.colorbar(im, cax=cax, **kwargs)       # draw cbar

def normalize_image(image, target_range=(0, 255)):
    """Normalize an image to a specified range."""
    min_val, max_val = np.min(image), np.max(image)
    if max_val > min_val:  # Avoid division by zero
        scaled_image = (image - min_val) / (max_val - min_val)  # Scale to [0, 1]
        return (scaled_image * (target_range[1] - target_range[0]) + target_range[0]).astype("uint8")
    else:
        return np.zeros_like(image, dtype="uint8")  # Handle constant images

def plot_layer_activations_with_anomalies(model, input_image, layer_indices,
                                          threshold=0.9):  # Define a threshold for "high activation"
    
    layer_summary = generate_layer_summary(model_summary)
    results_list = []
    for layer_index in layer_indices:
        # Create a sub-model for the specific layer
        intermediate_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
        activation = intermediate_model.predict(np.expand_dims(input_image, axis=0))  # Add batch dimension

        # Process activations
        if len(activation.shape) == 4:  # Batch, Height, Width, Channels
            activation_map = np.mean(activation[0], axis=-1)  # Average across channels for visualization
            activation_map_normalized = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map))  # Normalize

            # Highlight the region with the highest activation
            high_activation_mask = activation_map_normalized > threshold
            y_coords, x_coords = np.where(high_activation_mask)

            if y_coords.size > 0 and x_coords.size > 0:
                # Compute bounding box around the high activation region
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                # Rescale bounding box to match the input image dimensions
                input_h, input_w = input_image.shape[:2]
                activation_h, activation_w = activation_map.shape
                scale_x = input_w / activation_w
                scale_y = input_h / activation_h

                x_min, x_max = int(x_min * scale_x), int(x_max * scale_x)
                y_min, y_max = int(y_min * scale_y), int(y_max * scale_y)

                # Draw the bounding box on the original image
                input_image_with_box = input_image.copy()
                input_image_with_box = cv2.rectangle(
                    input_image_with_box,
                    (x_min, y_min),
                    (x_max, y_max),
                    color=(255, 0, 0),
                    thickness=2
                )

            else:
                input_image_with_box = input_image.copy()

            # Normalize input image for visualization
            input_image_normalized = normalize_image(input_image_with_box)

            # Plot the input image with bounding box and the activation map
            plt.figure(figsize=(6, 4))
            plt.suptitle(f"Layer {layer_index} Activation Analysis", y=0.95)

            # Input image with bounding box
            plt.subplot(1, 2, 1)
            plt.title(f"Input Image\nActivation > threshold ({threshold})",
                      fontsize=10)
            plt.imshow(input_image_normalized)
            # plt.axis("off")

            # Activation map
            plt.subplot(1, 2, 2)
            layer_description = layer_summary[layer_index] #.replace('_', '\n')
            plt.title(f"Layer {layer_index} Activation Map\n{layer_description}",
                      fontsize=10)
            im = plt.imshow(activation_map_normalized, cmap="viridis")
            add_colorbar(im)
            # plt.axis("off")

            plt.show()

            results_list.append({'layer': layer_index,
                                 'activation_map': activation_map_normalized})
        else:
            print(f"Layer {layer_index} output shape not supported for visualization: {activation.shape}")
    return results_list
