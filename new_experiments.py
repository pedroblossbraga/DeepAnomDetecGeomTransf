from multiprocessing import Process, Manager, freeze_support

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from keras.utils import to_categorical
import pandas as pd
import time
from datetime import datetime

from transformer import Transformer
from utils import load_cats_vs_dogs_tfds, save_dataset_tf, load_dataset_tf, get_class_name_from_index
from model import create_wide_residual_network
from performance import save_roc_pr_curve_data
from scores import compute_scores, compute_scores_with_entropy
from plots import plot_roc_auc_comparison

TODAYS_DT = datetime.now().strftime('%Y-%m-%d-%H%M')
OUTPUT_DIR = f'./results_experiments_{TODAYS_DT}_NoNewTransf_20epochs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_transformations_batch(x_train_task, y_train, transformer, single_class_ind):
  t_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
  x_train_task_transformed = transformer.transform_batch(
      np.repeat(x_train_task, transformer.n_transforms, axis=0), t_inds)
  return x_train_task_transformed, t_inds


def main_experiment(
        gpu_id, queue,
        
        include_new_transformations=False,
        save_dataset = False,
        load_dataset_online = True,
        load_precomputed=False,
        dataset_name='cats_vs_dogs',
        N_sample_train = 80,
        N_sample_test = 20,
        single_class_ind = 0,
        epochs=10
    ):
    np.random.seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    transf_img_dir = os.path.join(OUTPUT_DIR, 'transformed_images')
    os.makedirs(transf_img_dir, exist_ok=True)

    if load_dataset_online:
        print('loading dataset online...')
        # # Example Usage
        (x_train, y_train), (x_test, y_test) = load_cats_vs_dogs_tfds(img_size=(64, 64))

        if save_dataset:
            save_dataset_tf(
                tfrecord_path=OUTPUT_DIR,
                x=x_train,
                y=y_train,
                record_name='cats_vs_dogs_split80_train'
            )
            save_dataset_tf(
                tfrecord_path=OUTPUT_DIR,
                x=x_test,
                y=y_test,
                record_name='cats_vs_dogs_split80_test'
            )
    else:
        print('loading saved dataset')
        x_train, y_train = load_dataset_tf(
            os.path.join(OUTPUT_DIR, 'cats_vs_dogs_split80_train.tfrecord')
        )
        x_test, y_test = load_dataset_tf(
            os.path.join(OUTPUT_DIR, 'cats_vs_dogs_split80_test.tfrecord')
        )

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    ## instanciate geometric transformer
    transformer = Transformer(
        include_new_transformations=include_new_transformations,
        translation_x=16, translation_y=16,
        crop_size=(28, 28),
        output_size=(64, 64)
    )
    if f'cats_vs_dogs_data_split80_{int(N_sample_train)}.tfrecord' in os.listdir(OUTPUT_DIR) and load_precomputed:
        print('loading task images (task_train_features, task_train_labels)...')
        task_train_features, task_train_labels = load_dataset_tf(
            os.path.join(OUTPUT_DIR,  f'cats_vs_dogs_data_split80_{int(N_sample_train)}.tfrecord')
        )
    else:
        print('making task training data from scratch...')
        ## filter sample by class
        task_train_features = x_train[y_train.flatten() == single_class_ind]
        task_train_labels = y_train[y_train.flatten() == single_class_ind]
        
        ## sample training data (reducing due to computational limitations)
        print(f'samplig data with {N_sample_train} values')
        task_train_features = task_train_features[:N_sample_train]
        task_train_labels = task_train_labels[:N_sample_train]
    task_train_labels = task_train_labels.flatten()

    if f'cats_vs_dogs_data_split80_{int(N_sample_test)}_test.tfrecord' in os.listdir(OUTPUT_DIR) and load_precomputed:
        print('loading test sampled images (x_test, y_test)...')
        x_test, y_test = load_dataset_tf(
            os.path.join(OUTPUT_DIR,  f'cats_vs_dogs_data_split80_{int(N_sample_test)}_test.tfrecord')
        )
    else:
        print(f'samplig data with {N_sample_test} values')
        ## sample test data (reducing due to computational limitations)
        x_test = x_test[:N_sample_test]
        y_test = y_test[:N_sample_test]
    y_test = y_test.flatten()

    if save_dataset:
        # train
        save_dataset_tf(
            tfrecord_path=OUTPUT_DIR,
            x=task_train_features,
            y=task_train_labels,
            record_name=f'cats_vs_dogs_data_split80_{int(N_sample_train)}'
        )
        # test
        save_dataset_tf(
            tfrecord_path=OUTPUT_DIR,
            x=x_test,
            y=y_test,
            record_name=f'cats_vs_dogs_data_split80_{int(N_sample_test)}_test'
        )

    # Check and handle empty transformed data
    if f'cats_vs_dogs_data_split80_{int(N_sample_train)}_transf.tfrecord' in os.listdir(transf_img_dir):
        try:
            print('loading transformed images (x_train_task_transformed, t_inds)...')
            x_train_task_transformed, t_inds = load_dataset_tf(
                os.path.join(transf_img_dir, f'cats_vs_dogs_data_split80_{int(N_sample_train)}_transf.tfrecord')
            )
            if x_train_task_transformed.size == 0 or t_inds.size == 0:
                raise ValueError("Transformed data is empty.")
        except Exception as e:
            print(f"Error loading transformed data: {e}")
            print("Recomputing transformations...")
            x_train_task_transformed, t_inds = apply_transformations_batch(
                task_train_features,
                task_train_labels,
                transformer,
                single_class_ind
            )
    else:
        print("Transformed dataset not found. Generating...")
        x_train_task_transformed, t_inds = apply_transformations_batch(
            task_train_features,
            task_train_labels,
            transformer,
            single_class_ind
        )
    print('x_train_task_transformed shape: {}, t_inds shape: {}'.format(x_train_task_transformed.shape, t_inds.shape))
    
    
    # create model 
    n, k = (16, 8)
    model = create_wide_residual_network(
            x_train.shape[1:], transformer.n_transforms,
            n, k,
        )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('training model...')
    # train model
    model.fit(
        x_train_task_transformed,
        to_categorical(t_inds),
        batch_size=128,
        epochs=epochs
    )
    print('saving weights...')
    model.save_weights(os.path.join(OUTPUT_DIR, f'{dataset_name}_class_{single_class_ind}_{TODAYS_DT}.weights.h5'))

    t0=time.time()
    print('applying scores...')
    ## apply scores
    scores, predictions = compute_scores(model, x_test, transformer, task_train_features)
    dirichlet_time = time.time() - t0

    
    print('done applying scores!')
    print('scores shape: {}, predictions shape: {}'.format(scores.shape, predictions[0].shape))
    print('y_test shape: {}'.format(y_test.shape))

    try:
        np.savez_compressed(os.path.join(OUTPUT_DIR, 
            f'{dataset_name}_newtransf{str(include_new_transformations)}_class{single_class_ind}_results_{TODAYS_DT}.npz'),
                        scores=scores, predictions=predictions,
                        true_label=y_test.flatten(),
                        is_anomalous=y_test.flatten() != single_class_ind)
    except Exception as e:
        print('error saving results: {}'.format(e))

    # True labels for ROC AUC computation
    labels = y_test.flatten() == single_class_ind

    print('computing performance...')
    # save scores and results
    res_file_name = '{}_newtransf{}_class{}_{}.npz'.format(
        dataset_name, str(include_new_transformations),
        get_class_name_from_index(single_class_ind, dataset_name),
        TODAYS_DT
    )
    res_file_path = os.path.join(OUTPUT_DIR, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    t0 = time.time()
    print('applying score with entropy...')
    #### scores with entropy
    scores_entrop, predictions_entrop = compute_scores_with_entropy(model, x_test, transformer, task_train_features)
    print('done applying scores with entropy!')
    entropy_time = time.time() - t0

    print('scores_entrop shape: {}, predictions_entrop shape: {}'.format(scores_entrop.shape, predictions_entrop[0].shape))
    try:
        np.savez_compressed(os.path.join(OUTPUT_DIR, 
            f'{dataset_name}_newtransf{str(include_new_transformations)}_class{single_class_ind}_results_scores_entropy_{TODAYS_DT}.npz'),
                        scores_entrop=scores_entrop, predictions=predictions_entrop,
                        true_label=y_test.flatten(),
                        is_anomalous=y_test.flatten() != single_class_ind)
    except Exception as e:
        print('error saving results: {}'.format(e))

    print('computing performance...')
    res_file_name_entrop = '{}_newtransf{}_class{}_{}_entropy.npz'.format(
        dataset_name, str(include_new_transformations),
        get_class_name_from_index(single_class_ind, dataset_name),
        TODAYS_DT
    )
    res_file_path_entropy = os.path.join(OUTPUT_DIR, res_file_name_entrop)
    save_roc_pr_curve_data(scores_entrop, labels, res_file_path_entropy)
    
    try:
        plot_roc_auc_comparison(
            dirichlet_perf_path=res_file_path,
            entropy_perf_path=res_file_path_entropy,
            OUTPUT_DIR=OUTPUT_DIR
        )
    except Exception as e:
        print('error plotting roc auc comparison: {}'.format(e))


    pd.DataFrame({
        'dirichlet_time': [dirichlet_time],
        'entropy_time': [entropy_time]
    }).to_csv(os.path.join(OUTPUT_DIR, 'time_comparison.csv'), index=False)
    
    # Indicate GPU is free
    queue.put(gpu_id)

def run_gpu_manager(n_gpus,
    n_classes=1,
    dataset_name='cats_vs_dogs'
    ):
    os.makedirs(os.path.join(OUTPUT_DIR, dataset_name), exist_ok=True)


    manager = Manager()
    queue = manager.Queue(n_gpus)

    for g in range(n_gpus):
        queue.put(g)

    processes = []
    for single_class_ind in range(n_classes):
        gpu_id = queue.get()
        p = Process(target=main_experiment, 
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
    run_gpu_manager(N_GPUS)
    print('[Elapsed in time: {}]'.format(time.time() - t0))