import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
import os 
from datetime import datetime

from transformer import get_transformation_attributes

# Function to plot and save layer activations
def save_layer_activation(model, layer_index, input_image, file_path):
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    activation = intermediate_model.predict(np.expand_dims(input_image, axis=0))

    plt.figure(figsize=(10, 10))
    plt.title(f"Layer {layer_index} Activation")
    
    # plt.imshow(activation[0, :, :, 0], cmap='viridis')
    if len(activation.shape) == 4:  # Batch, Height, Width, Channels
        if activation.shape[-1] > 1:
            # Plot first channel
            plt.imshow(activation[0, :, :, 0], cmap='viridis')
        else:
            # Plot as a single-channel image
            plt.imshow(activation[0, :, :, 0], cmap='viridis')
    elif len(activation.shape) == 3:  # Batch, Height, Width
        plt.imshow(activation[0, :, :], cmap='viridis')
    elif len(activation.shape) == 2:  # Batch, Features
        plt.plot(activation[0])
    else:
        raise ValueError(f"Unsupported activation shape: {activation.shape}")


    plt.colorbar()
    plt.savefig(file_path)
    plt.close()

# Save example test images and their transformations
def save_test_images_with_transformations(transformer, x_test, y_test, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_examples = 3
    test_indices = np.random.choice(len(x_test), num_examples, replace=False)

    for i, idx in enumerate(test_indices):
        original_image = x_test[idx]
        true_label = y_test[idx]
        predicted_label = np.argmax(model.predict(np.expand_dims(original_image, axis=0)))

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.title(f"Original: {true_label}, Predicted: {predicted_label}")
        plt.imshow(original_image.squeeze(), cmap='gray')
        plt.axis('off')

        for j in range(3):
            transformed_image = transformer.transform_batch(np.expand_dims(original_image, axis=0), [j])[0]
            plt.subplot(1, 4, j + 2)
            plt.title(f"Transform {j + 1}")
            plt.imshow(transformed_image.squeeze(), cmap='gray')
            plt.axis('off')

        plt.savefig(os.path.join(output_dir, f"test_image_{i}.png"))
        plt.close()


def plot_first_three_images(x_train, y_train, dataset_name, class_names=None):
  """
  Plot the first three images in the dataset.

  Args:
      x_train (np.array): The training images.
      y_train (np.array): The training labels.
      class_names (list, optional): A list of class names corresponding to labels.
  """
  plt.figure(figsize=(15, 5))
  for i in range(3):
      plt.subplot(1, 3, i + 1)
      # Rescale image to [0, 1] for visualization
      image = (x_train[i] + 1) / 2.0
      plt.imshow(image)
      label = y_train[i]
      title = f"Label: {label}"
      if class_names:
          title += f" ({class_names[label]})"
      plt.title(title)
      plt.axis("off")
  plt.suptitle(f'Example images from {dataset_name}',
              y=1.03, fontsize=15)
  plt.tight_layout()
  plt.show()

def plot_all_transformations(transformer, original_image, transformation_list=None):
    """
    Plot all the transformations applied to the original image.
    """
    if transformation_list is None:
        transformation_list = transformer._transformation_list
    else:
        transformer._transformation_list = transformation_list

    # Number of transformations
    num_transformations = len(transformation_list)

    # Set up the grid for subplots
    cols = 4  # Number of columns
    rows = (num_transformations + cols - 1) // cols  # Calculate rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()  # Flatten the grid for easy indexing

    for i, transformation in enumerate(transformation_list):
        # Extract transformation parameters
        flip = transformation.flip
        tx = transformation.tx
        ty = transformation.ty
        k_rotate = transformation.k_90_rotate

        try:
          # Apply each transformation
          transformed_batch = transformer.transform_batch(
              np.expand_dims(original_image, axis=0), [i]  # Pass the index to apply the transformation
          )
          transformed_image = transformed_batch[0]  # Extract the single image from the batch
          transformed_image_vis = (transformed_image + 1) / 2.0  # Normalize for visualization

          # Display the image
          axes[i].imshow(transformed_image_vis)
          axes[i].set_title(f"Flip: {flip}, tx: {tx}, ty: {ty}, k: {k_rotate}")
          axes[i].axis('off')
        except Exception as e:
          print(f"Error applying {i}th transformation {get_transformation_attributes(i)}: {e}")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_roc_auc_comparison(
        dirichlet_perf_path,
        entropy_perf_path,
        OUTPUT_DIR
    ):
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    perf = np.load(dirichlet_perf_path)
    perf_entropy = np.load(entropy_perf_path)

    plt.figure(figsize=(6, 3))
    plt.title('Receiver Operating Characteristic')
    plt.plot(perf['fpr'], perf['tpr'], 'royalblue',
            #  linestyle='dashed',
            linewidth = 5,
            label = '(Dirichlet score) AUC = %0.2f' % perf['roc_auc'])
    plt.plot(perf_entropy['fpr'], perf_entropy['tpr'], 'darkorange',
            linestyle='dotted', linewidth = 5,
            label = '(Entropy score) AUC = %0.2f' % perf_entropy['roc_auc'])

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_auc_comparison.png'))
    # plt.show()