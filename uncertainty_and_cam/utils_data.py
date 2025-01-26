import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

def load_cats_vs_dogs_tfds(img_size=(64, 64)):
    """
    Load the Cats vs. Dogs dataset using TensorFlow Datasets, preprocess it,
    and split it into training and test sets.

    Args:
        img_size (tuple): Target size for resizing images (height, width).

    Returns:
        (x_train, y_train), (x_test, y_test): Preprocessed training and test data.
    """
    def preprocess_image(image, label):
        # Resize and normalize the image to [-1, 1]
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        return image, label

    # Load the dataset
    # train_split = 'train[:80%]'
    # test_split = 'train[80%:]'
    # train_ds = tfds.load('cats_vs_dogs', split=train_split, as_supervised=True)
    # test_ds = tfds.load('cats_vs_dogs', split=test_split, as_supervised=True)
    (train_ds, valid_ds, test_ds), info = tfds.load(
        'cats_vs_dogs',
        # take 50% for training, 25% for validation, and 25% for testing
        split=["train[:50%]", "train[50%:75%]", "train[75%:100%]"],
        as_supervised=True,
        with_info=True,
    )

    # Apply preprocessing
    train_ds = train_ds.map(preprocess_image)
    test_ds = test_ds.map(preprocess_image)

    # Convert to NumPy arrays
    x_train = []
    y_train = []
    for image, label in tfds.as_numpy(train_ds):
        x_train.append(image)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = []
    y_test = []
    for image, label in tfds.as_numpy(test_ds):
        x_test.append(image)
        y_test.append(label)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def save_dataset_tf(tfrecord_path,
                    record_name,
                    x, y=None,
                    dataset_name='cats_vs_dogs'):
    # Ensure the directory exists
    os.makedirs(tfrecord_path, exist_ok=True)

    print(f'saving {record_name} in {tfrecord_path}...')

    with tf.io.TFRecordWriter(os.path.ind_to_name[dataset_name][index](tfrecord_path, f'{record_name}.tfrecord')) as writer:
        if y is not None:
            for image, label in zip(x, y):
                # Convert image to uint8
                image_uint8 = tf.cast(tf.clip_by_value(image * 255.0, 0, 255), tf.uint8)
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[tf.io.encode_jpeg(image_uint8).numpy()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        else:
            for image in x:
                # Convert image to uint8
                image_uint8 = tf.cast(tf.clip_by_value(image * 255.0, 0, 255), tf.uint8)
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[tf.io.encode_jpeg(image_uint8).numpy()])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def _parse_function(proto):
    # Define the features you stored in the TFRecord
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),  # 'image' is stored as a byte string
        'label': tf.io.FixedLenFeature([1], tf.int64),  # 'label' is an int64 array with length 1
    }

    # Parse the example using the feature definition
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Decode the image data (JPEG encoded)
    parsed_features['image'] = tf.io.decode_jpeg(parsed_features['image'], channels=3)  # decode to 3 channels RGB

    return parsed_features['image'], parsed_features['label']

def load_dataset_tf(tfrecord_path):
    # Load the dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Parse the dataset
    dataset = dataset.map(_parse_function)

    # Initialize lists to store images and labels
    images_list = []
    labels_list = []

    # Iterate over the dataset and append images and labels to the lists
    for image, label in dataset:
        images_list.append(image.numpy())  # Convert the tensor to a NumPy array
        labels_list.append(label.numpy())  # Convert the tensor to a NumPy array

    # Convert lists to NumPy arrays
    x_train = np.array(images_list)
    y_train = np.array(labels_list)

    return x_train, y_train

def get_class_name_from_index(index, dataset_name):
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
        'cats_vs_dogs': ('cat', 'dog'),
    }

    return ind_to_name[dataset_name][index]

# N_sample_test = 20
# x_test, y_test = load_dataset_tf(
#             os.path.join(OUTPUT_DIR,  f'cats_vs_dogs_data_split80_{int(N_sample_test)}_test.tfrecord')
#         )