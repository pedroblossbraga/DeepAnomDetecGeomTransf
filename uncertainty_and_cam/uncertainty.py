import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import entropy

from plots import normalize_image

# Step 2: Monte Carlo Dropout for Uncertainty Estimation
def monte_carlo_uncertainty(model, inputs, num_passes=10):
    predictions = []
    for _ in range(num_passes):
        preds = model(inputs, training=True)  # Enable dropout during inference
        predictions.append(preds.numpy())
    predictions = np.array(predictions)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)
    return mean_predictions, uncertainty

def plot_predictions_with_uncertainty(
        # filename,
        OUTPUT_DIR,
        TODAYS_DT,
        images, predictions, uncertainty, labels=None):
    """
    Plot images with predictions and uncertainty as bar charts.
    Args:
        images: np.array of images.
        predictions: np.array of predicted labels or probabilities.
        uncertainty: np.array of uncertainties (variance).
        labels: List of true labels (optional).
    """
    transf_name = {
        0: "original",
        1: "flip",
        2: "rotate",
        3: "translate",
    }
    
    num_images = len(images)
    for i in range(num_images):
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        # plt.imshow(images[i].astype("uint8"))
        plt.imshow(normalize_image(images[i]))
        plt.title(f"Image {i+1}")
        if labels is not None:
            plt.xlabel(f"True Label: {labels[i]}")
        plt.axis("off")

        # Predictions Bar Chart
        plt.subplot(1, 3, 2)
        plt.bar(range(len(predictions[i])), predictions[i], color='blue')
        plt.title(f"Predictions")
        plt.xlabel("Transformation Classes")
        plt.ylabel("Probability")

        plt.xticks(range(len(uncertainty[i])),
                   [transf_name.get(i) for i in range(len(uncertainty[i]))],
                   rotation=-80)
        
        # Uncertainty Bar Chart
        plt.subplot(1, 3, 3)
        plt.bar(range(len(uncertainty[i])), uncertainty[i], color='orange')
        plt.title(f"Uncertainty")
        plt.xlabel("Transformation Classes")
        plt.ylabel("Variance")

        plt.xticks(range(len(uncertainty[i])),
                   [transf_name.get(i) for i in range(len(uncertainty[i]))],
                   rotation=-80)

        plt.tight_layout()
        # plt.savefig(filename)
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'preds_uncertainty_i{i}_{TODAYS_DT}.png')
        )
        # plt.show()
        plt.close()