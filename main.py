import numpy as np
import os
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Activation, Input, Conv2D, BatchNormalization, Flatten, Dense, Dropout, Reshape, \
    AveragePooling2D, Add
from keras.regularizers import l2
import tensorflow as tf
from multiprocessing import Pool, cpu_count
import itertools
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy.special import psi, polygamma

import pandas as pd
from utils import save_roc_pr_curve_data, get_class_name_from_index

import shutil
# Ensure reproducibility of dataset download
dataset_cache_dir = os.path.expanduser("~/.keras/datasets")
shutil.rmtree(dataset_cache_dir, ignore_errors=True)

# Ensure output directory exists
OUTPUT_DIR = 'cpu_version_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enable XLA for faster training
tf.config.optimizer.set_jit(True)

# Dataset loaders
def normalize_minus1_1(data):
    return 2 * (data / 255.) - 1

def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = tf.cast(X_train, dtype=tf.float32)  # Cast to float32
    X_test = tf.cast(X_test, dtype=tf.float32)    # Cast to float32
    X_train = normalize_minus1_1(tf.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant'))
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = normalize_minus1_1(tf.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant'))
    X_test = np.expand_dims(X_test, axis=-1)
    return (X_train, y_train), (X_test, y_test)

# Wide Residual Network
def create_wide_residual_network(input_shape, num_classes, depth, widen_factor=1):
    def batch_norm(): return BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
    def conv2d(out_channels, kernel_size, strides=1): return Conv2D(out_channels, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.0005))
    def dense(out_units): return Dense(out_units, kernel_regularizer=l2(0.0005))

    def add_basic_block(x_in, out_channels, strides):
        is_channels_equal = x_in.shape[-1] == out_channels
        bn1 = batch_norm()(x_in)
        bn1 = Activation('relu')(bn1)
        out = conv2d(out_channels, 3, strides)(bn1)
        out = batch_norm()(out)
        out = Activation('relu')(out)
        out = conv2d(out_channels, 3, 1)(out)
        shortcut = x_in if is_channels_equal else conv2d(out_channels, 1, strides)(bn1)
        return Add()([out, shortcut])

    def add_conv_group(x_in, out_channels, n, strides):
        out = add_basic_block(x_in, out_channels, strides)
        for _ in range(1, n):
            out = add_basic_block(out, out_channels, 1)
        return out

    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    n = (depth - 4) // 6

    inp = Input(shape=input_shape)
    conv1 = conv2d(n_channels[0], 3)(inp)
    conv2 = add_conv_group(conv1, n_channels[1], n, 1)
    conv3 = add_conv_group(conv2, n_channels[2], n, 2)
    conv4 = add_conv_group(conv3, n_channels[3], n, 2)

    out = batch_norm()(conv4)
    out = Activation('relu')(out)
    out = AveragePooling2D(8)(out)
    out = Flatten()(out)
    out = dense(num_classes)(out)
    out = Activation('softmax')(out)
    return Model(inp, out)

# Transformations
class AffineTransformation:
    def __init__(self, flip, tx, ty, k_90_rotate, scale=1.0, shear=0, output_size=(32, 32)):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate
        self.scale = scale
        self.shear = shear
        self.output_size = output_size

    def __call__(self, x):
        res_x = np.fliplr(x) if self.flip else x
        res_x = tf.keras.preprocessing.image.apply_affine_transform(
            res_x, tx=self.tx, ty=self.ty, zx=self.scale, zy=self.scale, shear=self.shear, fill_mode='reflect'
        )
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)
        return res_x

def get_transformation_attributes(t_idx):
    T = Transformer()._transformation_list[t_idx]
    attrs = vars(T)
    return str(T) + ', '.join("%s: %s" % item for item in attrs.items())

class Transformer:
    def __init__(self, translation_x=8, translation_y=8):
        self.max_tx = translation_x
        self.max_ty = translation_y
        self._transformation_list = []
        self._create_transformation_list()

    def _create_transformation_list(self):
        for flip, tx, ty, k_rotate, scale, shear in itertools.product(
            [False, True],
            [0, -self.max_tx, self.max_tx],
            [0, -self.max_ty, self.max_ty],
            range(4),
            [1.0, 1.2],
            [0, 15]
        ):
            self._transformation_list.append(
                AffineTransformation(flip=flip, tx=tx, ty=ty, k_90_rotate=k_rotate, scale=scale, shear=shear)
            )

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    def transform_batch(self, x_batch, t_inds):
        transformed_batch = x_batch.copy()
        N_t_inds = len(t_inds)
        for i, t_ind in enumerate(t_inds):
            print('({}/{}) ({:.3f}%)'.format(
                i+1, N_t_inds, (i+1)*100/N_t_inds
            ))
            # print(f'transformation index: {t_ind}')
            transformed_batch[i] = self._transformation_list[t_ind](transformed_batch[i])
            print(f'successfully applied T {t_ind}: {get_transformation_attributes(t_ind)}')
        return transformed_batch

# Dirichlet Score Computation
def inv_psi(y, iters=5):
    cond = y >= -2.22
    x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))
    for _ in range(iters):
        x = x - (psi(x) - y) / polygamma(1, x)
    return x

def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
    alpha_new = alpha_old = alpha_init
    for _ in range(max_iter):
        alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
        if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
            break
        alpha_old = alpha_new
    return alpha_new

def dirichlet_normality_score(alpha, p):
    return np.sum((alpha - 1) * np.log(p), axis=-1)

def compute_scores(model, x_test, transformer, x_train_task):
    scores = np.zeros((len(x_test),))
    for t_ind in range(transformer.n_transforms):
        observed_dirichlet = model.predict(transformer.transform_batch(x_train_task, [t_ind] * len(x_train_task)), batch_size=128)
        observed_dirichlet = np.clip(observed_dirichlet, 1e-10, 1.0)
        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

        alpha_sum_approx = len(observed_dirichlet) * (len(observed_dirichlet[0]) - 1) * (-psi(1))
        alpha_sum_approx /= len(observed_dirichlet) * np.sum(observed_dirichlet * np.log(observed_dirichlet)) - np.sum(observed_dirichlet * np.sum(np.log(observed_dirichlet), axis=0))
        alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)
        x_test_p = model.predict(transformer.transform_batch(x_test, [t_ind] * len(x_test)), batch_size=128)
        x_test_p = np.clip(x_test_p, 1e-10, 1.0)
        scores += dirichlet_normality_score(mle_alpha_t, x_test_p)
    scores /= transformer.n_transforms
    return scores

# Parallel Training
def run_training_for_class(args):
    dataset_loader, dataset_name, single_class_ind = args
    (x_train, y_train), (x_test, y_test) = dataset_loader()

    if dataset_name in ['cats-vs-dogs']:
        transformer = Transformer(16, 16)
        n, k = (16, 8)
    else:
        transformer = Transformer(8, 8)
        n, k = (10, 4)

    model = create_wide_residual_network(x_train.shape[1:], transformer.n_transforms, n, k)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    x_train_task = x_train[y_train.flatten() == single_class_ind]
    transformations_inds = np.tile(np.arange(transformer.n_transforms), len(x_train_task))
    x_train_task_transformed = transformer.transform_batch(np.repeat(x_train_task, transformer.n_transforms, axis=0),
                                                           transformations_inds)
    # Training
    batch_size = 128
    model.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
            batch_size=batch_size, epochs=int(np.ceil(200 / transformer.n_transforms))
            )
    model.save_weights(os.path.join(OUTPUT_DIR, dataset_name, f'{dataset_name}_class_{single_class_ind}_weights.h5'))

    # Evaluate
    scores = compute_scores(model, x_test, transformer, x_train_task)
    labels = y_test.flatten() == single_class_ind

    # compute current ROC AUC
    roc_auc = roc_auc_score(labels, scores[:, single_class_ind])

    # Optionally, save ROC and PR curve data for future visualization
    res_file_name = '{}_transformations_{}_{}.npz'.format(
        dataset_name,
        get_class_name_from_index(single_class_ind, dataset_name),
        datetime.now().strftime('%Y-%m-%d-%H%M')
    )
    res_file_path = os.path.join(OUTPUT_DIR, dataset_name, res_file_name)
    save_roc_pr_curve_data(scores, labels, res_file_path)

    return single_class_ind, roc_auc


def cpu_parallel_training(dataset_loader, dataset_name, n_classes):
    # ensure output folder exists
    os.makedirs(os.path.join(OUTPUT_DIR, dataset_name), exist_ok=True)

    n_workers = min(cpu_count(), 8)  # Use 8 CPUs max
    args = [(dataset_loader, dataset_name, single_class_ind) for single_class_ind in range(n_classes)]
    with Pool(n_workers) as pool:
        results = pool.map(run_training_for_class, args)
    for single_class_ind, roc_auc in results:
        print(f"Class {single_class_ind} ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    cpu_parallel_training(load_fashion_mnist, "fashion-mnist", 10)
