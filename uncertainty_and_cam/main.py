import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go

import pandas as pd
import numpy as np
import os

import time
from datetime import datetime

from multiprocessing import Process, Manager, freeze_support

from keras.utils import to_categorical

# from google.colab import drive
# # Mount Google Drive
# drive.mount('/content/drive', force_remount=True)
# OUTPUT_DIR = 'drive/MyDrive/FAU - masters/semester 3/Anomaly Detection/new_results'
# os.listdir(OUTPUT_DIR)
TODAYS_DT = datetime.now().strftime('%Y-%m-%d-%H%M')
experiment_suffix = '' #'_zoom'
OUTPUT_DIR = f'./results_experiments_{TODAYS_DT}{experiment_suffix}'
os.makedirs(OUTPUT_DIR, exist_ok=True)


from utils_data import load_cats_vs_dogs_tfds
from transformer import Transformer, AffineTransformation
from model import create_wide_residual_network
from uncertainty import monte_carlo_uncertainty, plot_predictions_with_uncertainty
from gradcam import grad_cam, overlay_heatmap, plot_all_gradcams
from normality_scores import compute_scores_with_entropy, save_top_images_with_scores, compute_scores
from performance import save_roc_pr_curve_data, plot_roc_auc_comparison
# from scores import compute_scores, compute_scores_with_entropy


def apply_transformations_batch(x_train_task, y_train, transformer, single_class_ind):
  t_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0), t_inds)
  return x_train_task_transformed, t_inds


def run_experiment(
        gpu_id, queue,
        epochs=10,
        dataset_name='cats_vs_dogs',
        N_sample_train = 80,
):
    np.random.seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # extract dataset
    (x_train, y_train), (x_test, y_test) = load_cats_vs_dogs_tfds(img_size=(64, 64))
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # selected sample
    cat_images = [
        x_test[y_test == 0][7],
        x_test[y_test == 0][11],
        x_test[y_test == 0][12]
    ]
    dog_images = [
        x_test[y_test == 1][7],
        x_test[y_test == 1][11],
        x_test[y_test == 1][12]
    ]
    sample_test = np.array(cat_images + dog_images)

    sample_labels = np.array([0] * len(cat_images) + [1] * len(dog_images))

    # defining the image transformer
    transformer = Transformer(
        include_new_transformations=False,
        translation_x=16, translation_y=16,
        crop_size=(28, 28),
        output_size=(64, 64)
    )
    transformation_list = [
            # original
            AffineTransformation(
                flip=False, tx=0, ty=0, k_90_rotate=0,
                zoom=1.,
                crop_size=None,
                color_jitter=False,
                histogram_eq=False,
                output_size=(64, 64)
            ),
            # flip
            AffineTransformation(
                flip=True, tx=0, ty=0, k_90_rotate=0,
                zoom=1.,
                crop_size=None,
                color_jitter=False,
                histogram_eq=False,
                output_size=(64, 64)
            ),
            # rotate
            AffineTransformation(
                flip=False, tx=0, ty=0, k_90_rotate=3,
                zoom=1.,
                crop_size=None,
                color_jitter=False,
                histogram_eq=False,
                output_size=(64, 64)
            ),
            # translate
            AffineTransformation(
                flip=False, tx=16, ty=16, k_90_rotate=0,
                zoom=1.,
                crop_size=None,
                color_jitter=False,
                histogram_eq=False,
                output_size=(64, 64)
            ),
    ]
    transformer._transformation_list = transformation_list

    # Prepare Transformed Training Data
    single_class_ind = 0  # Class index for in-class samples (e.g., 'cat')
    task_train_features = x_train[y_train.flatten() == single_class_ind]
    task_train_labels = y_train[y_train.flatten() == single_class_ind]

    ## sample training data (reducing due to computational limitations)
    print(f'samplig data with {N_sample_train} values')
    task_train_features = task_train_features[:N_sample_train]
    task_train_labels = task_train_labels[:N_sample_train]

    x_train_task_transformed, t_inds = apply_transformations_batch(
                task_train_features,
                task_train_labels,
                transformer,
                single_class_ind
            )
    print('x_train_task_transformed shape: {}, t_inds shape: {}'.format(x_train_task_transformed.shape, t_inds.shape))
    
    ############################################################
    # create model 
    n, k = (16, 8)
    new_model = create_wide_residual_network(
            x_train.shape[1:], transformer.n_transforms,
            n, k,
        )
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('training model...')
    # train model
    new_model.fit(
        x_train_task_transformed,
        to_categorical(t_inds),
        batch_size=128,
        # epochs=int(np.ceil(200 / transformer.n_transforms))
        # epochs=200
        epochs=epochs
    )
    print('saving weights...')
    new_model.save_weights(os.path.join(OUTPUT_DIR, 
                                        f'{dataset_name}_class_{single_class_ind}_{TODAYS_DT}.weights.h5'))

    ##############################################################
    x_test_transformed, t_inds_test = apply_transformations_batch(
        sample_test, sample_labels,
        transformer, single_class_ind
    )

    ##############################################################
    # Apply Monte Carlo Dropout to test data
    num_passes = 50
    mean_preds, uncertainty = monte_carlo_uncertainty(
        new_model, x_test_transformed, num_passes)

    print(f"Uncertainty shape: {uncertainty.shape}")

    plot_predictions_with_uncertainty(
        OUTPUT_DIR=OUTPUT_DIR,
        TODAYS_DT=TODAYS_DT,
        images=x_test_transformed, 
        predictions=mean_preds, 
        uncertainty=uncertainty, 
        labels=None
    )

    ##############################################################
    # Visualize Grad-CAM for the first test sample
    # image = x_test_transformed[:1]  # Select a single transformed test sample
    #     class_idx = np.argmax(mean_preds[0])  # Predicted transformation class
    # last_conv_layer_name = "conv2d_15"  # Update with your model's last conv layer name (index 45)

    # heatmap = grad_cam(new_model, image, class_idx, last_conv_layer_name)
    # overlay = overlay_heatmap(image[0], heatmap)

    # plt.imshow(overlay)
    # plt.title("Grad-CAM Overlay")
    # plt.axis("off")
    # plt.savefig(os.path.join(OUTPUT_DIR, f'gradcam_{TODAYS_DT}.png'))
    # # plt.show()
    # plt.close()

    plot_all_gradcams(
        model=new_model,
        images=x_test_transformed,
        mean_preds=mean_preds,
        last_conv_layer_name="conv2d_15",  # Update with your model's last conv layer name
        output_dir=os.path.join(OUTPUT_DIR, f'gradcams_{TODAYS_DT}'),
        labels=sample_labels
    )

    ##############################################################
    # Apply normality scoring to the test set
    normality_scores, predictions = compute_scores_with_entropy(
        new_model,
        # x_test, 
        sample_test,
        transformer, task_train_features)
    print(f"Normality scores shape: {normality_scores.shape}")

    sns.displot(normality_scores,
                kind='kde', fill=True)
    plt.savefig(os.path.join(OUTPUT_DIR,f'dist_normality_scores_{TODAYS_DT}.png'))
    # plt.show()
    plt.close()

    print("Top 10 normality scores:", normality_scores[:10])

    # Save the top 10 normality score images
    save_top_images_with_scores(
        images=sample_test, 
        scores=normality_scores, 
        labels=sample_labels, 
        output_dir=os.path.join(OUTPUT_DIR, "top_scores")
    )

    # Step 5: Combine Predictions, Uncertainty, and Scores
    # def combine_scores(mean_preds, uncertainty, normality_scores):
    #     combined_scores = normality_scores - np.mean(uncertainty, axis=1)  # Penalize high uncertainty
    #     return combined_scores

    def combine_scores(mean_preds, uncertainty, normality_scores, n_transforms):
        """
        Combine normality scores with uncertainty to produce a final score.

        Args:
            mean_preds (np.ndarray): Mean predictions for test samples (e.g., (24, 4)).
            uncertainty (np.ndarray): Uncertainty values (e.g., (24, 4)).
            normality_scores (np.ndarray): Normality scores for the original samples (e.g., (6,)).
            n_transforms (int): Number of transformations applied per test sample.

        Returns:
            np.ndarray: Combined scores for all test samples.
        """
        num_original_samples = len(normality_scores)

        # Debugging output
        print(f"Mean predictions shape: {mean_preds.shape}")
        print(f"Uncertainty shape: {uncertainty.shape}")
        print(f"Normality scores shape: {normality_scores.shape}")
        print(f"Number of transforms: {n_transforms}")
        print(f"Number of original samples: {num_original_samples}")

        # Ensure `uncertainty` dimensions align
        if uncertainty.shape[0] != num_original_samples * n_transforms:
            raise ValueError(
                f"Cannot reshape uncertainty array of size {uncertainty.size} "
                f"into shape ({num_original_samples}, {n_transforms}). "
                f"Ensure consistency between number of test samples, transformations, and uncertainty array."
            )

        # Aggregate uncertainty across transforms (average over transforms)
        uncertainty = uncertainty.reshape(num_original_samples, n_transforms, -1)  # Reshape by samples and transforms
        aggregated_uncertainty = np.mean(uncertainty, axis=1)  # Aggregate across transforms (shape: (6, 4))

        # Final combined score: penalize high uncertainty
        combined_scores = normality_scores[:, np.newaxis] - np.mean(aggregated_uncertainty, axis=1, keepdims=True)
        return combined_scores.flatten()



    print(f"Uncertainty array shape: {uncertainty.shape}")
    print(f"Number of transforms: {transformer.n_transforms}")
    print(f"Number of original samples (normality scores): {len(normality_scores)}")

    combined_scores = combine_scores(mean_preds, uncertainty, normality_scores, transformer.n_transforms)

    # Step 6: Analyze Results
    # Plot the distribution of scores
    plt.hist(combined_scores, bins=30, alpha=0.7, label="Combined Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Normality Scores with Uncertainty Penalization")
    plt.savefig(os.path.join(OUTPUT_DIR,f'combined_scores.png'))
    # plt.show()
    plt.close()


    # True labels for ROC AUC computation
    ground_truth_labels = sample_labels.flatten() == single_class_ind

    # compute dirichlet and entropy scores    
    scores_dirichlet, predictions = compute_scores(new_model, sample_test, transformer, task_train_features)
    scores_entrop, predictions = compute_scores_with_entropy(new_model, sample_test, transformer, task_train_features)

    # Save ROC and PR curve data
    perf_dirichlet = save_roc_pr_curve_data(
        scores=scores_dirichlet,
        labels=ground_truth_labels,  # Replace with actual ground truth
        file_path=os.path.join(OUTPUT_DIR, f"perf_dirichlet_{TODAYS_DT}.npz")
    )
    perf_dirichlet['method']='dirichlet'

    perf_entropy = save_roc_pr_curve_data(
        scores=scores_entrop,
        labels=ground_truth_labels,  # Replace with actual ground truth
        file_path=os.path.join(OUTPUT_DIR, f"perf_entropy_{TODAYS_DT}.npz")
    )
    perf_entropy['method']='entropy'

    perf_entropy_uncertainty = save_roc_pr_curve_data(
        scores=combined_scores,
        labels=ground_truth_labels,  # Replace with actual ground truth
        file_path=os.path.join(OUTPUT_DIR, f"perf_entropy_uncert_{TODAYS_DT}.npz")
    )
    perf_entropy_uncertainty['method']='entropy_uncertainty'

    perfs = pd.DataFrame([perf_dirichlet, perf_entropy, perf_entropy_uncertainty])
    perfs.to_csv(os.path.join(OUTPUT_DIR, f'perfs_{TODAYS_DT}.csv'))

    plot_roc_auc_comparison(
        perf_dirichlet,
        perf_entropy,
        perf_entropy_uncertainty,
        OUTPUT_DIR
    )


def main(n_gpus,
    n_classes=1,
    # dataset_name='cats_vs_dogs'
    ):
    manager = Manager()
    queue = manager.Queue(n_gpus)

    for g in range(n_gpus):
        queue.put(g)

    processes = []
    for _ in range(n_classes):
        gpu_id = queue.get()
        p = Process(target=run_experiment, 
                    args=(gpu_id, queue, 
        ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    t0 = time.time()
    freeze_support()
    N_GPUS = 1
    # N_GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
    main(N_GPUS)
    print('[Elapsed in time: {}]'.format(time.time() - t0))