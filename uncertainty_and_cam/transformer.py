import tensorflow as tf

from skimage.transform import resize
from skimage import exposure
import itertools
import os, sys

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

def hist_eq(I, quantile=0.6):
    """
    Perform histogram equalization with reduced effect by limiting the adjustment
    to a given quantile of the image histogram.

    Parameters:
    - I: Input image (array-like), expected to be in the range [0, 255].
    - quantile: Float, upper limit of intensity quantile to stretch to 255.

    Returns:
    - I_equalized: Equalized image with limited intensity adjustment.
    """
    # Handle different input scales
    if np.min(I) < 0 or np.max(I) <= 1:
      # Rescale from [-1, 1] or [0, 1] to [0, 255]
      I = ((I - np.min(I)) / (np.max(I) - np.min(I))) * 255

    # Ensure image values are between 0 and 255 and convert to uint8
    I = np.clip(I, 0, 255).astype(np.uint8)

    print('Image range: {} to {}'.format(
        np.min(I), np.max(I)
    ))

    # Fetch histogram and cumulative distribution function (CDF)
    hist, bins = np.histogram(I.flatten(), bins=256, range=(0, 255), density=True)
    cdf = hist.cumsum()

    # Normalize the CDF to [0, 1]
    cdf_normalized = cdf / cdf[-1]

    # Determine the intensity range to adjust based on quantile
    lower_limit = 0  # Always stretch from 0
    upper_limit = np.searchsorted(cdf_normalized, quantile)

    # Scale the CDF only within the specified range
    cdf_scaled = np.interp(range(256),
                    [lower_limit, upper_limit],
                    [0, 255]).astype(np.uint8)

    # Apply the adjusted CDF to the image
    I_equalized = cdf_scaled[I]

    # rescale back to [-1,1]
    I = ((I - np.min(I)) / (np.max(I) - np.min(I))) -1

    return I_equalized

def get_transformation_attributes(t_idx):
    T = Transformer()._transformation_list[t_idx]
    attrs = vars(T)
    return str(T) + ', '.join("%s: %s" % item for item in attrs.items())

class AffineTransformation:
    def __init__(self, flip, tx, ty, k_90_rotate,
                #  scale=1.0,
                 zoom = 1.0,
                #  shear=0,
                 crop_size=None,
                 color_jitter=False, histogram_eq=False,
                 hist_eq_quantile=0.7,
                 output_size=(32, 32)):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate

        # new transformations
        # self.scale = scale
        self.zoom = zoom
        # self.shear = shear
        self.crop_size = crop_size
        self.color_jitter = color_jitter
        self.histogram_eq = histogram_eq

        self.hist_eq_quantile = hist_eq_quantile

        self.output_size = output_size  # Target output size

    def __call__(self, x):
        print(f"Initial image shape: {x.shape}")
        res_x = x.copy()

        if x.shape[0] == 3:  # Assuming CHW format
            res_x = np.moveaxis(res_x, 0, -1)

        if self.flip:
            res_x = np.fliplr(res_x)
            print(f"Initial image (before transformations): {res_x[0, 0, :]}")  # Show the first pixel

        if self.tx != 0 or self.ty != 0:
            # Clamp tx and ty to valid ranges
            image_height, image_width = res_x.shape[:2]
            tx = max(-image_width, min(self.tx, image_width))
            ty = max(-image_height, min(self.ty, image_height))

            res_x = tf.keras.preprocessing.image.apply_affine_transform(
                res_x, tx=tx, ty=ty,
                row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='reflect',
                # shear=self.shear,
                )
            print(f"Translation applied: tx={self.tx}, ty={self.ty}")
            print(f"After translation (first pixel): {res_x[0, 0, :]}")

        # Zoom effect
        if self.zoom != 1.0:
            zoom_factor = 1 / self.zoom  # Reciprocal since we're zooming into the image
            crop_h = int(res_x.shape[0] * zoom_factor)
            crop_w = int(res_x.shape[1] * zoom_factor)
            center_h, center_w = res_x.shape[0] // 2, res_x.shape[1] // 2

            top = max(0, center_h - crop_h // 2)
            left = max(0, center_w - crop_w // 2)

            cropped = res_x[top:top + crop_h, left:left + crop_w]
            print(f"Zoom applied: zoom_factor={self.zoom}, cropped_shape={cropped.shape}")

            # Resize back to output size
            res_x = resize(cropped, self.output_size, anti_aliasing=True, preserve_range=True)
            print(f"Resized to output size: {self.output_size}")

        # Rotations
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)
            print(f"Rotation applied: k_rotate={self.k_90_rotate}")
            print(f"After rotation (first pixel): {res_x[0, 0, :]}")

        # Random Crop
        if self.crop_size:
            crop_h, crop_w = self.crop_size
            h, w = x.shape[:2]
            if h >= crop_h and w >= crop_w:
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
                res_x = res_x[top:top + crop_h, left:left + crop_w]

        # Color Jitter
        if self.color_jitter:
            res_x = res_x * (np.random.uniform(0.8, 1.2))  # Random brightness adjustment
            res_x = np.clip(res_x, 0, 255)

        # Histogram Equalization
        if self.histogram_eq:
            # # Normalize to [0, 1] if input is in range [-1, 1]
            # res_x = (res_x + 1) / 2 if np.min(res_x) < 0 else res_x

            # # Apply histogram equalization
            # # res_x = exposure.equalize_hist(res_x)
            # # Apply adaptive histogram equalization with clip limit
            # res_x = exposure.equalize_adapthist(res_x / 255.0, clip_limit=0.8)  # Clip limit to reduce over-enhancement

            # # Scale to [0, 255]
            # res_x = res_x * 255
            # res_x = np.clip(res_x, 0, 255)  # Ensure valid range

            # # Handle different input scales
            # if np.min(res_x) < 0 or np.max(res_x) <= 1:
            #   # Rescale from [-1, 1] or [0, 1] to [0, 255]
            #   res_x = ((res_x - np.min(res_x)) / (np.max(res_x) - np.min(res_x))) * 255

            # Ensure image values are between 0 and 255 and convert to uint8
            # res_x = np.clip(res_x, 0, 255).astype(np.uint8)

            res_x = hist_eq(res_x, quantile=self.hist_eq_quantile)
            # res_x = np.clip(res_x, 0, 255).astype(np.uint8)
            # res_x = np.clip(res_x, -1, 1).astype(np.uint8)
            res_x = (res_x - 127.5)/127.5

            print("Histogram Equalization applied")
            print(f"After histogram equalization range: {res_x.min()}, {res_x.max()}")
            print(f"Final image (first pixel): {res_x[0, 0, :]}")

        # Resize to ensure consistent output shape
        res_x = resize(res_x, self.output_size, anti_aliasing=True, preserve_range=True)

        print(f"Final shape: {res_x.shape}")  # Debugging: print final image shape

        return res_x

def get_transformation_attributes(t_idx):
    T = Transformer()._transformation_list[t_idx]
    attrs = vars(T)
    return str(T) + ', '.join("%s: %s" % item for item in attrs.items())

class Transformer:
    def __init__(self,
                 include_new_transformations=False,
                 translation_x=16, translation_y=16,
                 crop_size=(28, 28),
                 hist_eq_quantile=0.7,
                 output_size=(64, 64)
                 ):
        self.include_new_transformations = include_new_transformations
        self.max_tx = translation_x
        self.max_ty = translation_y
        self.crop_size = crop_size
        self._transformation_list = []  # Initialize the list
        self.output_size = output_size
        self.hist_eq_quantile = hist_eq_quantile

        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)  # Ensure this works correctly

    def _create_transformation_list(self):
      transformation_list = []
      if self.include_new_transformations:

        for flip, tx, ty, k_rotate, zoom, color_jitter, hist_eq in itertools.product(
          # for flip, tx, ty, k_rotate, crop_size, shear, color_jitter, hist_eq in itertools.product(
            [False, True],
             [0, -self.max_tx, self.max_tx],
              [0, -self.max_ty, self.max_ty],
            range(4),
            [1.0, 1.2],
            # [None, (20, 20)],
            # [0, 15],
             [False, True],
             [False, True]
        ):
            self._transformation_list.append(
                AffineTransformation(
                    flip=flip, tx=tx, ty=ty, k_90_rotate=k_rotate,
                    zoom=zoom,
                    # shear=shear,
                    # crop_size=self.crop_size if np.random.rand() > 0.5 else None,
                    # crop_size=crop_size,
                    color_jitter=color_jitter,
                    histogram_eq=hist_eq,
                    hist_eq_quantile=self.hist_eq_quantile
                )
            )

      else:
        transformation_list = itertools.product(
            [False, True],
             [0, -self.max_tx, self.max_tx],
              [0, -self.max_ty, self.max_ty],
               range(4)
        )
        # transformation_list = list(transformation_list)
        # print(f"transformation_list: {(transformation_list)}")

        for flip, tx, ty, k_rotate in transformation_list:
          self._transformation_list.append(
              AffineTransformation(
                  flip=flip, tx=tx, ty=ty, k_90_rotate=k_rotate,
                  output_size = self.output_size
              )
          )

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)
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

        zoom = transformation.zoom
        # shear = transformation.shear
        crop_size = transformation.crop_size
        color_jitter = transformation.color_jitter
        histogram_eq = transformation.histogram_eq
        hist_eq_quantile = transformation.hist_eq_quantile

        try:
          # Apply each transformation
          transformed_batch = transformer.transform_batch(
              np.expand_dims(original_image, axis=0), [i]  # Pass the index to apply the transformation
          )
          transformed_image = transformed_batch[0]  # Extract the single image from the batch
          transformed_image_vis = (transformed_image + 1) / 2.0  # Normalize for visualization

          # Display the image
          axes[i].imshow(transformed_image_vis)
          if histogram_eq:
            histogram_eq = f'{histogram_eq} (Q: {hist_eq_quantile})'

          axes[i].set_title(f"Flip: {flip}, tx: {tx}, ty: {ty}, k: {k_rotate}\n"+\
                            # f"Zoom: {zoom}, Shear: {shear}, Crop: {crop_size}\n"+\
                            f"Zoom: {zoom}, Crop: {crop_size}\n"+\
                            f"Color Jitter: {color_jitter}, Histogram Eq: {histogram_eq}"
                            )
          axes[i].axis('off')
        except Exception as e:
          print(f"Error applying {i} transformation {get_transformation_attributes(i)}: {e}")
          exc_type, exc_obj, exc_tb = sys.exc_info()
          fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
          print('exc type: {}, fname: {}, lineno: {}'.format(exc_type, fname, exc_tb.tb_lineno))

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()