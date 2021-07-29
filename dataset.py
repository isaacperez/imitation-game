import glob
import os
import random
import csv
import time

import tensorflow as tf
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

import CONST

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DATASET_PATH = "../dataset/"

TRAIN_CSV_FILE = "train"
VAL_CSV_FILE = "val"
TEST_CSV_FILE = "test"

CSV_FILE_EXTENSION = ".csv"

PATH_COL_IDX_IN_CSV = 0
PATH_CLASS_IDX_IN_CSV = 1

MINIMUM_NUMBER_OF_EXAMPLES_TRAIN = 20
NUMBER_OF_EXAMPLES_VAL = 200
NUMBER_OF_EXAMPLES_TEST = 300

DATA_TYPE_INPUT = tf.dtypes.uint8
ANOMALOUS_SIZE = 500

# Data augmentation
BRIGHTNESS = "brightness"
MIN_VAL_BRIGHTNESS = -0.10
MAX_VAL_BRIGHTNESS = 0.20
BRIGHTNESS_DEFAULT = 0.15

CONTRAST = "contrast"
MIN_VAL_CONTRAST = 0.80
MAX_VAL_CONTRAST = 1.20
CONTRAST_DEFAULT = 0.15

GAMMA = "gamma"
MIN_GAMMA_VALUE_GAMMA = 1.05
MAX_GAMMA_VALUE_GAMMA = 1.25
MIN_GAIN_VALUE_GAMMA = 1.0
MAX_GAIN_VALUE_GAMMA = 1.0
GAMMA_DEFAULT = 0.1

HUE = "hue"
MIN_VAL_HUE = -0.08
MAX_VAL_HUE = 0.08
HUE_DEFAULT = 0.15

SATURATION = "saturation"
MIN_VAL_SATURATION = 0.7
MAX_VAL_SATURATION = 1.5
SATURATION_DEFAULT = 0.15

GAUSSIAN_NOISE = "gaussian_noise"
GAUSSIAN_NOISE_DEFAULT = 0.15

FLIP_VERTICAL = "flip_vertical"
FLIP_VERTICAL_DEFAULT = 0.5

FLIP_HORIZONTAL = "flip_horizontal"
FLIP_HORIZONTAL_DEFAULT = 0.15

ROTATION = "rotation"
ROTATION_MIN_VALUE = -40
ROTATION_MAX_VALUE = 40
ROTATION_DEFAULT = 0.0

BLUR = "blur"
BLUR_FACTOR = 5
BLUR_DEFAULT = 0.15

SALT_AND_PEPPER = "salt_and_pepper"
SALT_AND_PEPPER_RATIO_FACTOR = 0.005
SALT_AND_PEPPER_DEFAULT = 0.9

SHEAR = "shear"
SHEAR_FACTOR = 40
SHEAR_DEFAULT = 0.0

EXPECTED_PROBABILITIES_KEYS = {BRIGHTNESS, CONTRAST, GAMMA, HUE, SATURATION,
                               GAUSSIAN_NOISE, FLIP_VERTICAL, ROTATION,
                               FLIP_HORIZONTAL, BLUR, SALT_AND_PEPPER,
                               SHEAR}

# imgaug probailities
rotation_prob = 0.6
affine_prob = 0.6
flip_lr_prob = 0.5
flip_ud_prob = 0.0

arithmetic_prob = 0.6
arithmetic_same_time_ops = 2

cutout_prob = 0.6
quality_prob = 0.6

# Define the imgaug transformations
seq = iaa.Sequential([
        iaa.Sometimes(rotation_prob, iaa.Affine(rotate=(-10, 10))),
        iaa.Sometimes(affine_prob, iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, 
                                      translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                      shear=(-16, 16))),
        iaa.Fliplr(flip_lr_prob),
        iaa.Flipud(flip_ud_prob),
        iaa.Sometimes(arithmetic_prob, iaa.SomeOf(arithmetic_same_time_ops, [
            iaa.Add((-20, 20), per_channel=0.5),
            iaa.Multiply((0.7, 1.3), per_channel=0.5),
            iaa.AdditiveGaussianNoise(scale=0.05*255)], random_order=True)),
        iaa.Sometimes(cutout_prob, iaa.SomeOf(1, [
            iaa.Cutout(fill_mode="constant", cval=(0, 255),
                fill_per_channel=0.5, nb_iterations=(1, 4), size=(0.05,0.1)),
            iaa.Cutout(fill_mode="gaussian", cval=(0, 255),
                fill_per_channel=0.5, nb_iterations=(1, 4), size=(0.05,0.1)),
            iaa.CoarseDropout(0.02, size_percent=0.5),
            iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
            iaa.SaltAndPepper(0.05),
            iaa.SaltAndPepper(0.05, per_channel=True),
            iaa.CoarseSaltAndPepper(0.02, size_percent=(0.01, 0.05), 
                per_channel=True)])),
        iaa.Sometimes(quality_prob, iaa.SomeOf(1, [
            iaa.JpegCompression(compression=(70, 99)),
            iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0)),
            iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(5)),
            iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(20)),
            iaa.GaussianBlur(sigma=(0.0, 1.5)),
            iaa.MedianBlur(k=(3, 5)),
            iaa.MotionBlur(k=7),
            iaa.WithHueAndSaturation(
                iaa.WithChannels(0, iaa.Add((0, 20)))),
            iaa.MultiplyHue((0.85, 1.25)),
            iaa.MultiplySaturation((0.75, 1.35)),
            iaa.ChangeColorTemperature((3000, 10000)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.CLAHE(clip_limit=(1, 10)),
            iaa.AllChannelsHistogramEqualization(),
            iaa.Sharpen(alpha=(0.05, 0.2), lightness=(0.75, 1.0)),
            iaa.Emboss(alpha=(0.3, 0.6), strength=(0.5, 1.5)),
            iaa.AveragePooling(2),
            iaa.MedianPooling(2),
            iaa.Clouds(),
            iaa.Rain(speed=(0.1, 0.3))
            ]))
    ])


# -----------------------------------------------------------------------------
# Local functions
# -----------------------------------------------------------------------------
def get_train_PIC_elements():
    """Read the CSV TRAIN_FILE in DATASET_PATH"""

    with open(DATASET_PATH + TRAIN_CSV_FILE + CSV_FILE_EXTENSION) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        path_list = []
        class_list = []
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                path_list.append(row[PATH_COL_IDX_IN_CSV])
                class_list.append(row[PATH_CLASS_IDX_IN_CSV])

            line_count += 1

        idx_list = list(range(len(path_list)))
    
    return path_list, idx_list, class_list


def get_test_PIC_elements():
    """Read the CSV TEST_CSV_FILE in DATASET_PATH"""

    with open(DATASET_PATH + TEST_CSV_FILE + CSV_FILE_EXTENSION) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        path_list = []
        class_list = []
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                path_list.append(row[PATH_COL_IDX_IN_CSV])
                class_list.append(row[PATH_CLASS_IDX_IN_CSV])

            line_count += 1

        idx_list = list(range(len(path_list)))
    
    return path_list, idx_list, class_list


def get_val_PIC_elements():
    """Read the CSV VAL_CSV_FILE in DATASET_PATH"""

    with open(DATASET_PATH + VAL_CSV_FILE + CSV_FILE_EXTENSION) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        path_list = []
        class_list = []
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                path_list.append(row[PATH_COL_IDX_IN_CSV])
                class_list.append(row[PATH_CLASS_IDX_IN_CSV])

            line_count += 1

        idx_list = list(range(len(path_list)))
    
    return path_list, idx_list, class_list


def pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst):
    """This function transform the input list to the PIC format 
       (each path is transformed in [path, index, Class])

        Keyword arguments:
        lst -- list of paths

        Restrictions:
        lst must not have duplicated elements
    """
    assert len(set(path_lst)) == len(path_lst), "There are duplicated path-elements" + str(len(idx_lst))
    assert len(set(idx_lst)) == len(idx_lst), "There are duplicated idx-elements"
    assert np.min(idx_lst) == 0 and np.max(idx_lst) == len(idx_lst)-1, "Bad range of idx-elements"
    assert len(path_lst) == len(label_lst), "path_lst and label_lst have different number of elements"
    assert len(path_lst) == len(idx_lst), "path_lst and idx_lst have different number of elements"

    # Built the PIC list from lst
    PIC_lst = [[path_lst[i], idx_lst[i], CONST.CLASSES.index(label_lst[i])] for i in range(len(path_lst))]

    return PIC_lst


def replicate_PIC_list_by_number_of_replication(PIC_lst, replications_by_elements):
    """This function replicate each element in the input list with the PIC format
       as times as is indicate in the same position in replications_by_elements argument.

        Keyword arguments:
        PIC_lst -- list with PIC format to be replicated
        replications_by_elements -- num of replications by position

        Restrictions:
        PIC_lst must not have duplicated elements and must have PIC format
        replications factor must be greather than 0
    """

    assert len(set([PIC_lst[i][0] for i in range(len(PIC_lst))])) == len(PIC_lst), \
        "There are duplicated elements"

    assert all([x > 0 for x in replications_by_elements]), \
        "There are replications factor less or equal than 0"

    each_element_replicate_in_a_list = \
        [[element] * replications_by_elements[idx] for idx, element in enumerate(PIC_lst)]

    each_element_replicate = \
        [item for items in each_element_replicate_in_a_list for item in items]

    return each_element_replicate


def get_sample_by_class_from_PIC_list(pic, number_of_samples_by_class):
    """This function obtain a random sub-pic list from pic where each class have
       numberOfSamplesByClass elements by class.

        Keyword arguments:
        pic -- [path, index, Class] list
        number_of_samples_by_class -- num of elements in each class
    """
    auxiliar = [random.sample([x for x in pic if x[2] == i], number_of_samples_by_class) for i in
                range(len(CONST.CLASSES))]
    subpic = [item for items in auxiliar for item in items]

    return subpic


# -----------------------------------------------------------------------------
# Module functions: EXTRACT
# -----------------------------------------------------------------------------
def parse_train(path_input, idx_src, idx_class):
    """Get and return the tensors after reading the data based on the
    path_input for the train dataset and its one-hot class encoding.

    Keyword arguments:
    path_input -- list with the paths of images
    idx_src -- idx of the current path in the originial list
    idx_class -- the class index
    """

    # Reading the data from the disk
    img_raw = tf.io.read_file(path_input)

    # Formatting the byte array to expected shapes
    img_decode = tf.io.decode_image(img_raw, channels=CONST.NUM_CHANNELS_INPUT)

    return img_decode, idx_src, tf.one_hot(idx_class, len(CONST.CLASSES))


def parse_val(path_input, idx_src, idx_class):
    """Get and return the tensors after reading the data based on the
    path_input for the val dataset and its one-hot class encoding.

    Keyword arguments:
    path_input -- list with the paths of images
    idx_src -- idx of the current path in the originial list
    idx_class -- the class index
    """

    # Reading the data from the disk
    img_raw = tf.io.read_file(path_input)

    # Formatting the byte array to expected shapes
    img_decode = tf.io.decode_image(img_raw, channels=CONST.NUM_CHANNELS_INPUT)

    return img_decode, idx_src, tf.one_hot(idx_class, len(CONST.CLASSES))


def parse_test(path_input, idx_src, idx_class):
    """Get and return the tensors after reading the data based on the
    path_input for the test dataset and its one-hot class encoding.

    Keyword arguments:
    path_input -- list with the paths of images
    idx_src -- idx of the current path in the originial list
    idx_class -- the class index
    """

    # Reading the data from the disk
    img_raw = tf.io.read_file(path_input)

    # Formatting the byte array to expected shapes
    img_decode = tf.io.decode_image(img_raw, channels=CONST.NUM_CHANNELS_INPUT)

    return img_decode, idx_src, tf.one_hot(idx_class, len(CONST.CLASSES))


# -----------------------------------------------------------------------------
# Module functions: TRANSFORM
# -----------------------------------------------------------------------------
def preprocess_train(img_decode, idx_src, idx_class_one_hot):
    """Get and return the tensors with the proper shape and type from the
    original tensor for images and keep the other data as it is.

    Keyword arguments:
    img_decode -- tensor of image/s
    idx_src -- idx of the image in the original path list
    class_one_hot -- expected class with one-hot encoding
    """
    # Convert images to float
    img = tf.image.convert_image_dtype(img_decode, tf.float32)

    img = tf.image.resize_with_pad(img,
                                   target_height=CONST.HIGH_SIZE,
                                   target_width=CONST.WIDTH_SIZE,
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   antialias=False)

    return img, idx_src, idx_class_one_hot


def preprocess_val(img_decode, idx_src, idx_class_one_hot):
    """Get and return the tensors with the proper shape and type from the
    original tensor for images and keep the other data as it is.

    Keyword arguments:
    img_decode -- tensor of image/s
    idx_src -- idx of the image in the original path list
    class_one_hot -- expected class with one-hot encoding
    """
    # Convert images to float
    img = tf.image.convert_image_dtype(img_decode, tf.float32)
    img = tf.image.resize_with_pad(img,
                                   target_height=CONST.HIGH_SIZE,
                                   target_width=CONST.WIDTH_SIZE,
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   antialias=False)

    return img, idx_src, idx_class_one_hot


def preprocess_test(img_decode, idx_src, idx_class_one_hot):
    """Get and return the tensors with the proper shape and type from the
    original tensor for images and keep the other data as it is.

    Keyword arguments:
    img_decode -- tensor of image/s
    idx_src -- idx of the image in the original path list
    class_one_hot -- expected class with one-hot encoding
    """
    # Convert images to float
    img = tf.image.convert_image_dtype(img_decode, tf.float32)
    img = tf.image.resize_with_pad(img,
                                   target_height=CONST.HIGH_SIZE,
                                   target_width=CONST.WIDTH_SIZE,
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   antialias=False)

    return img, idx_src, idx_class_one_hot


def data_augmentation(img, idx_src, idx_class_one_hot, probabilities):
    """Get and return the tensors with the proper shape and type from the
    original tensor for images in train dataset and keep the other
    data as it is.

    Keyword arguments:
    img -- tensor of one image
    probabilities -- dict with probabilities € [0,1] of applying each data
    augmentation technique.
    """

    # Checks probabilities before use
    assert all(0.0 <= value <= 1.0 for key, value in probabilities.items()), \
        "Probabilities must have in [0.0, 1.0]"

    assert probabilities.keys() == EXPECTED_PROBABILITIES_KEYS, \
        "Probabilities must have only these keys: " \
        + str(EXPECTED_PROBABILITIES_KEYS)
    
    # Hue
    img = tf.cond(tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                          probabilities[HUE]),
                  lambda: tf.image.adjust_hue(img, tf.random.uniform(
                     shape=[], minval=MIN_VAL_HUE,
                     maxval=MAX_VAL_HUE)),
                  lambda: img)

    # Saturation
    img = tf.cond(tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                          probabilities[SATURATION]),
                  lambda: tf.image.adjust_saturation(img, tf.random.uniform(
                   shape=[], minval=MIN_VAL_SATURATION,
                   maxval=MAX_VAL_SATURATION)),
                  lambda: img)

    # Brightness
    appy_brightness = tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                              probabilities[BRIGHTNESS])
    img = tf.cond(appy_brightness,
                  lambda: tf.image.adjust_brightness(img, tf.random.uniform(
                      shape=[], minval=MIN_VAL_BRIGHTNESS,
                      maxval=MAX_VAL_BRIGHTNESS)),
                  lambda: img)

    # Contrast
    appy_contrast = tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                            probabilities[CONTRAST])
    img = tf.cond(appy_contrast,
                  lambda: tf.image.adjust_contrast(img, tf.random.uniform(
                     shape=[], minval=MIN_VAL_CONTRAST,
                     maxval=MAX_VAL_CONTRAST)),
                  lambda: img)

    # Gamma
    apply_gamma = tf.math.logical_not(tf.math.logical_or(appy_brightness,
                                                         appy_contrast))
    img = tf.cond(tf.math.logical_and(tf.less(tf.random.uniform(shape=[],
                                                                minval=0,
                                                                maxval=1),
                                              probabilities[GAMMA]),
                                      apply_gamma),
                  lambda: tf.image.adjust_gamma(img, tf.random.uniform(
                     shape=[], minval=MIN_GAMMA_VALUE_GAMMA,
                     maxval=MAX_GAMMA_VALUE_GAMMA), tf.random.uniform(
                     shape=[], minval=MIN_GAIN_VALUE_GAMMA,
                     maxval=MAX_GAIN_VALUE_GAMMA)),
                  lambda: img)
    
    # Gaussian Noise
    img = tf.cond(tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                          probabilities[GAUSSIAN_NOISE]),
                  lambda: img + tf.random.normal(tf.shape(img), mean=0.0,
                                                 stddev=0.08,
                                                 dtype=tf.dtypes.float32),
                  lambda: img)
    
    # Flip vertical
    apply_flip_vertical = tf.less(tf.random.uniform(shape=[], minval=0,
                                                    maxval=1),
                                  probabilities[FLIP_VERTICAL])

    img = tf.cond(apply_flip_vertical, lambda: tf.image.flip_up_down(img),
                  lambda: img)

    # Flip horizontal
    apply_flip_horizontal = tf.less(tf.random.uniform(shape=[], minval=0,
                                                      maxval=1),
                                    probabilities[FLIP_HORIZONTAL])

    img = tf.cond(apply_flip_horizontal, lambda: tf.image.flip_left_right(img),
                  lambda: img)
    
    # Blur
    img = tf.cond(tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                          probabilities[BLUR]),
                  lambda: tf.squeeze(tf.nn.avg_pool2d(input=tf.expand_dims(img, 0), ksize=BLUR_FACTOR, strides=1, padding='SAME')),
                  lambda: img)

    # Limit pixel values to [0, 1]
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)

    # Salt & Pepper
    mask = tf.random.uniform(shape=[int(CONST.HIGH_SIZE / 8), int(CONST.WIDTH_SIZE / 8)], minval=0.0, maxval=1.0, dtype=tf.float32)
    mask = tf.cast(mask > SALT_AND_PEPPER_RATIO_FACTOR, tf.dtypes.uint8)
    mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, 3])

    mask = tf.nn.max_pool2d(tf.expand_dims(1 - mask, 0), ksize=(3, 3), strides=1, padding='SAME')
    mask = tf.squeeze(mask)
    mask = 1 - mask
    mask = tf.image.resize(mask, size=[CONST.HIGH_SIZE, CONST.WIDTH_SIZE], method='nearest', antialias=False)

    mask = tf.cast(mask, tf.dtypes.float32)

    img = tf.cond(tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),
                          probabilities[SALT_AND_PEPPER]),
                  lambda: img * mask,
                  lambda: img)

    """
    # The following transformations work with the numpy object
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rotation
    apply_rotation = tf.less(tf.random.uniform(shape=[], minval=0,
                                               maxval=1),
                             probabilities[ROTATION])

    img = tf.cond(apply_rotation,
                  lambda: tf.keras.preprocessing.image.random_rotation(
                            x=img, rg=ROTATION_ANGLE, row_axis=0, col_axis=1, channel_axis=2),
                  lambda: img)

    # Shear
    apply_shear = tf.less(tf.random.uniform(shape=[], minval=0,
                                               maxval=1),
                             probabilities[SHEAR])

    img = tf.cond(apply_shear,
                  lambda: tf.keras.preprocessing.image.random_shear(
                            x=img, intensity=SHEAR_FACTOR, row_axis=0, col_axis=1, channel_axis=2),
                  lambda: img) 

    # Transform to tensor before return the image
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    """

    return img, idx_src, idx_class_one_hot


def ETL_imgaug(path_input, idx_src, idx_class):
    """Do the whole ETL training process with the imgaug library.
    
    Keyword arguments:
    path_input -- list with the paths of images
    idx_src -- idx of the current path in the originial list
    idx_class -- the class index
    """

    # Reading the data from the disk (path_input is a tensor originally so we transform it to Python string)
    image = imageio.imread(path_input.numpy().decode("utf-8"))

    # Resize the image
    image = tf.cast(tf.image.resize_with_pad(tf.cast(image, tf.dtypes.float32),
                                   target_height=CONST.HIGH_SIZE,
                                   target_width=CONST.WIDTH_SIZE,
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   antialias=False), tf.dtypes.uint8).numpy()

    # Apply random transformation
    image_aug = seq(image=image)

    # Convert images to TensorFlow float32 tensor with values in [0, 1]
    image_aug = tf.convert_to_tensor(image_aug, dtype=tf.dtypes.float32) / 255.0

    # Get he one-hot encoding
    one_hot_encoding = tf.one_hot(idx_class, len(CONST.CLASSES))

    return image_aug, idx_src, one_hot_encoding


# -----------------------------------------------------------------------------
# Module functions: LOAD
# -----------------------------------------------------------------------------
def build_train_dataset(batch_size, num_batches_preloaded, num_parallel_calls,
                        allow_repetitions, probabilities, replications, shuffle=True, 
                        use_tensorflow_data_aug=False):
    """Combine all functions to build train dataset

    Keyword arguments:
    batch_size -- batch size of training
    num_batches_preloaded -- num batches in memory
    num_parallel_calls -- num threads to work with dataset
    allow_repetitions -- allow to repeat batch in the dataset
    probabilities -- dict of probabilities for each data aumentation technique
    replications -- replication of each path
    shuffle -- do random permutations
    use_tensorflow_data_aug -- data augmentation with Tensorflow Ops or imgaug library
    """
    # Get the paths
    path_lst, idx_lst, label_lst = get_train_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)
    
    # Get the minimum number of element for all classes
    class_idx = [element[2] for element in pic]
    minimum_number_of_elements = min([class_idx.count(idx) for idx in range(len(CONST.CLASSES))])

    # Replicate the paths
    pic = replicate_PIC_list_by_number_of_replication(pic, replications)

    # Sample by the minimum number of elements
    subpic = get_sample_by_class_from_PIC_list(pic, minimum_number_of_elements)

    # Split each component
    paths = [element[0] for element in subpic]
    idx_srcs = [element[1] for element in subpic]
    idx_classes = [element[2] for element in subpic]

    # Build the dataset from the list of paths
    train_dataset = tf.data.Dataset.from_tensor_slices((paths,
                                                        idx_srcs,
                                                        idx_classes))

    # Shuffle the dataset by the number of examples
    if shuffle:
        train_dataset = train_dataset.shuffle(len(subpic))

    # Allow repetition to do more than one epoch
    if allow_repetitions:
        train_dataset = train_dataset.repeat()

    # Define the ETL proces by the data augmentation library to use
    if use_tensorflow_data_aug:
        # Now we can process the paths. First extract phase
        train_dataset = train_dataset.map(parse_train,
                                        num_parallel_calls=num_parallel_calls)

        # Now we can process the paths. Second transform phase
        train_dataset = train_dataset.map(preprocess_train,
                                        num_parallel_calls=num_parallel_calls)

        # Apply data aumentation function
        def data_aumentation_function(img, idx_src, idx_class_one_hot):
            return data_augmentation(img, idx_src, idx_class_one_hot, probabilities)

        train_dataset = train_dataset.map(data_aumentation_function,
                                        num_parallel_calls=num_parallel_calls)

    else:

        # Do the ETL process with the imgaug library
        train_dataset = train_dataset.map(lambda paths, idx_srcs, idx_classes: \
                                            tf.py_function(ETL_imgaug, [paths, idx_srcs, idx_classes], [tf.float32, tf.int32, tf.float32]),
                                          num_parallel_calls=num_parallel_calls)

    # Tensorflow need to know the data shape to define the graph properly
    def define_final_shape(img, idx, one_hot):
        return tf.reshape(img, shape=(CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT)), idx, one_hot
    
    train_dataset = train_dataset.map(define_final_shape)

    # Set the number of examples for each batch
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    # Set the number of batch loaded in memory
    train_dataset = train_dataset.prefetch(num_batches_preloaded)

    return train_dataset


def build_val_dataset(batch_size, num_batches_preloaded, num_parallel_calls):
    """Combine all functions to build val dataset

    Keyword arguments:
    batch_size -- batch size of val
    num_batches_preloaded -- num batches in memory
    num_parallel_calls -- num threads to work with dataset
    """
    # Get the paths
    path_lst, idx_lst, label_lst = get_val_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

 
    # Split each component
    paths = [element[0] for element in pic]
    idx_srcs = [element[1] for element in pic]
    idx_classes = [element[2] for element in pic]

    # Build the dataset from the list of paths
    val_dataset = tf.data.Dataset.from_tensor_slices((paths,
                                                        idx_srcs,
                                                        idx_classes))
    
    # Now we can process the paths. First extract phase
    val_dataset = val_dataset.map(parse_val,
                                      num_parallel_calls=num_parallel_calls)

    # Now we can process the paths. Second transform phase
    val_dataset = val_dataset.map(preprocess_val,
                                      num_parallel_calls=num_parallel_calls)

    def define_final_shape(img, idx, one_hot):
        return tf.reshape(img, shape=(CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT)), idx, one_hot
    
    val_dataset = val_dataset.map(define_final_shape)

    # Set the number of examples for each batch
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    # Set the number of batch loaded in memory
    val_dataset = val_dataset.prefetch(num_batches_preloaded)
    
    return val_dataset


def build_test_dataset(batch_size, num_batches_preloaded, num_parallel_calls):
    """Combine all functions to build test dataset

    Keyword arguments:
    batch_size -- batch size of test
    num_batches_preloaded -- num batches in memory
    num_parallel_calls -- num threads to work with dataset
    """
    # Get the paths
    path_lst, idx_lst, label_lst = get_test_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)
 
    # Split each component
    paths = [element[0] for element in pic]
    idx_srcs = [element[1] for element in pic]
    idx_classes = [element[2] for element in pic]

    # Build the dataset from the list of paths
    test_dataset = tf.data.Dataset.from_tensor_slices((paths,
                                                        idx_srcs,
                                                        idx_classes))
    
    # Now we can process the paths. First extract phase
    test_dataset = test_dataset.map(parse_test,
                                      num_parallel_calls=num_parallel_calls)

    # Now we can process the paths. Second transform phase
    test_dataset = test_dataset.map(preprocess_test,
                                      num_parallel_calls=num_parallel_calls)

    # Set the number of examples for each batch
    test_dataset = test_dataset.batch(batch_size)

    # Set the number of batch loaded in memory
    test_dataset = test_dataset.prefetch(num_batches_preloaded)
    
    return test_dataset


# -----------------------------------------------------------------------------
# Module functions: UTILS
# -----------------------------------------------------------------------------
def to_human_output(img, one_hot_encoding):
    """Transform the dataset format data to human interpretable format"""
    return img, CONST.CLASSES[tf.argmax(one_hot_encoding)]


def data_augmentation_calibration(probabilities):
    """Show the result of the data_augmentation for the defined probabilities"""

    # Get all the paths
    path_lst, idx_lst, label_lst  = get_train_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        img_decode, idx_src_new, idx_class_one_hot = parse_train(path, idx_src, idx_class)

        img_transformed, idx_src_new, idx_class_one_hot = \
            preprocess_train(img_decode, idx_src_new, idx_class_one_hot)

        img, idx_src_new, idx_class_one_hot_new = data_augmentation(img_transformed, idx_src_new, idx_class_one_hot,
                                                                    probabilities)

        img, class_human = to_human_output(img, idx_class_one_hot_new)

        print("Image", i, "of", NUM_DATA, ":", path)

        plt.figure(1)
        plt.title("Original image (B = "
                  + str(tf.round(tf.reduce_mean(img_transformed, axis=(0, 1)) * 255, 2).numpy())
                  + "): " + class_human)
        plt.imshow(img_transformed)
        plt.draw()

        plt.figure(2)
        plt.title("Augmented image (B = "
                  + str(tf.round(tf.reduce_mean(img, axis=(0, 1)) * 255, 2).numpy())
                  + "): " + class_human)
        plt.imshow(img)
        plt.draw()

        plt.show(block=False)

        # Wait break only with a keyboard press
        while not plt.waitforbuttonpress():
            pass

    plt.close()


def build_probabilities_dict(brightness=BRIGHTNESS_DEFAULT, contrast=CONTRAST_DEFAULT,
                             gamma=GAMMA_DEFAULT, hue=HUE_DEFAULT, saturation=SATURATION_DEFAULT,
                             gaussian_noise=GAUSSIAN_NOISE_DEFAULT, blur=BLUR_DEFAULT,
                             flip_vertical=FLIP_VERTICAL_DEFAULT, flip_horizontal=FLIP_HORIZONTAL_DEFAULT,
                             rotation=ROTATION_DEFAULT, salt_and_pepper=SALT_AND_PEPPER_DEFAULT,
                             shear=SHEAR_DEFAULT):
    """Build the dict of probabilities of the data augmentation params"""

    probabilities = {BRIGHTNESS: brightness, CONTRAST: contrast, GAMMA: gamma, HUE: hue,
                     SATURATION: saturation, GAUSSIAN_NOISE: gaussian_noise, BLUR: blur,
                     FLIP_VERTICAL: flip_vertical, FLIP_HORIZONTAL: flip_horizontal,
                     ROTATION: rotation, SALT_AND_PEPPER: salt_and_pepper, SHEAR: shear}

    assert all(0.0 <= value <= 1.0 for key, value in probabilities.items()), \
        "Probabilities must have in [0.0, 1.0]"

    assert probabilities.keys() == EXPECTED_PROBABILITIES_KEYS, \
        "Probabilities must have only these keys: " \
        + str(probabilities.keys()) + "- " + str(EXPECTED_PROBABILITIES_KEYS)

    return probabilities


def get_basic_training_variables(batch_size):
    """Returns all the variables needed during the training"""

    # We need the data augmentation probabilities dic
    probabilities = build_probabilities_dict()

    # We need the paths to warn when we find a dificult image
    path_lst, idx_lst, label_lst = get_train_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    # We need to modify the replication factor of each element by its classification result
    replications = [1 for _ in range(len(path_lst))]

    # We need the number of batches in the train dataset to print it
    class_idx = [element[2] for element in pic]
    minimum_number_of_elements = min([class_idx.count(idx) for idx in range(len(CONST.CLASSES))])
    batches_in_the_train_dataset = int(np.floor((minimum_number_of_elements * len(CONST.CLASSES)) / batch_size))

    return probabilities, path_lst, replications, batches_in_the_train_dataset


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_get_train_PIC_elements():
    """Test for get_train_paths()"""

    # Call the function
    train_paths, train_idx, train_labels  = get_train_PIC_elements()

    # Test the length
    assert len(train_paths) > 0, "There are no paths in the train set"
    assert len(train_paths) == len(train_labels), "There are different number of paths than labels"
    assert len(train_paths) == len(train_idx), "There are different number of paths than idx"

    # Test the length of paths by classes
    paths_by_classes = \
        [[path for path in train_paths if path.split(os.sep)[-2] == cls] for cls in CONST.CLASSES]
    assert all([len(cls_path) >= MINIMUM_NUMBER_OF_EXAMPLES_TRAIN for cls_path in paths_by_classes]), \
        "There are no paths for a/some class/classes: " \
        + str([i for i in list(zip(CONST.CLASSES, [len(i) for i in paths_by_classes])) if
               i[1] < MINIMUM_NUMBER_OF_EXAMPLES_TRAIN])

    # Test the length of labels by classes
    labels_by_classes = \
        [[label for label in train_labels if label == cls] for cls in CONST.CLASSES]
    assert all([len(cls_label) >= MINIMUM_NUMBER_OF_EXAMPLES_TRAIN for cls_label in labels_by_classes]), \
        "There are no label for a/some class/classes: " \
        + str([i for i in list(zip(CONST.CLASSES, [len(i) for i in paths_by_classes])) if
               i[1] < MINIMUM_NUMBER_OF_EXAMPLES_TRAIN])
    
    # Test the file extension
    file_extensions = ["." + path.split(".")[-1] for path in train_paths]
    assert all([file_extension in CONST.IMG_EXTENSION for file_extension in file_extensions]), \
        "There are some files with non valid extension: " \
        + str([path for path in train_paths if "." + path.split(".")[-1] not in CONST.IMG_EXTENSION])

    # Test the elements are unique
    assert len(set(train_paths)) == len(train_paths), \
        "There are elements duplicated"


def test_get_test_PIC_elements():
    """Test for get_test_paths()"""

    # Call the function
    test_paths, test_idx, test_labels  = get_test_PIC_elements()

    # Test the length
    assert len(test_paths) > 0, "There are no paths in the test set"
    assert len(test_paths) == len(test_labels), "There are different number of paths than labels"
    assert len(test_paths) == len(test_idx), "There are different number of paths than idx"


    # Test the length of paths by classes
    paths_by_classes = \
        [[path for path in test_paths if path.split(os.sep)[-2] == cls]   for cls in CONST.CLASSES]
    assert all([len(cls_path) >= NUMBER_OF_EXAMPLES_TEST for cls_path in paths_by_classes]), \
        "There are no paths for a/some class/classes: " \
        + str([i for i in list(zip(CONST.CLASSES, [len(i) for i in paths_by_classes])) if i[1] < NUMBER_OF_EXAMPLES_TEST])

    # Test the length of labels by classes
    labels_by_classes = \
        [[label for label in test_labels if label == cls]   for cls in CONST.CLASSES]
    assert all([len(cls_label) >= NUMBER_OF_EXAMPLES_TEST for cls_label in labels_by_classes]), \
        "There are no label for a/some class/classes: " \
        + str([i for i in list(zip(CONST.CLASSES, [len(i) for i in paths_by_classes])) if i[1] < NUMBER_OF_EXAMPLES_TEST])

    # Test the file extension
    file_extensions = ["." + path.split(".")[-1] for path in test_paths]
    assert all([file_extension in CONST.IMG_EXTENSION for file_extension in file_extensions]), \
        "There are some files with non valid extension: " \
        + str([path for path in test_paths if "." + path.split(".")[-1] not in CONST.IMG_EXTENSION])

    # Test the elements are unique
    assert len(set(test_paths)) == len(test_paths), \
        "There are elements duplicated"


def test_get_val_PIC_elements():
    """Test for get_test_paths()"""

    # Call the function
    val_paths, val_idx, val_labels  = get_test_PIC_elements()

    # Test the length
    assert len(val_paths) > 0, "There are no paths in the val set"
    assert len(val_paths) == len(val_labels), "There are different number of paths than labels"
    assert len(val_paths) == len(val_idx), "There are different number of paths than idx"

    # Test the length of paths by classes
    paths_by_classes = \
        [[path for path in val_paths if path.split(os.sep)[-2] == cls]   for cls in CONST.CLASSES]
    assert all([len(cls_path) >= NUMBER_OF_EXAMPLES_VAL for cls_path in paths_by_classes]), \
        "There are no paths for a/some class/classes: " \
        + str([i for i in list(zip(CONST.CLASSES, [len(i) for i in paths_by_classes])) if i[1] < NUMBER_OF_EXAMPLES_VAL])

    # Test the length of labels by classes
    labels_by_classes = \
        [[label for label in val_labels if label == cls]   for cls in CONST.CLASSES]
    assert all([len(cls_label) >= NUMBER_OF_EXAMPLES_VAL for cls_label in labels_by_classes]), \
        "There are no label for a/some class/classes: " \
        + str([i for i in list(zip(CONST.CLASSES, [len(i) for i in paths_by_classes])) if i[1] < NUMBER_OF_EXAMPLES_VAL])

    # Test the file extension
    file_extensions = ["." + path.split(".")[-1] for path in val_paths]
    assert all([file_extension in CONST.IMG_EXTENSION for file_extension in file_extensions]), \
        "There are some files with non valid extension: " \
        + str([path for path in val_paths if "." + path.split(".")[-1] not in CONST.IMG_EXTENSION])

    # Test the elements are unique
    assert len(set(val_paths)) == len(val_paths), \
        "There are elements duplicated"


def test_pathlabelidx_lists_to_PIC_format():
    """Test for pathlabelidx_lists_to_PIC_formatt()"""

    # Call the function
    paths_lst, idx_lst, label_lst  = get_val_PIC_elements()

    PIC_lst = pathlabelidx_lists_to_PIC_format(paths_lst, label_lst, idx_lst)

    # Test the length
    assert len(paths_lst) == len(PIC_lst), " len(output) != len(input)"


    # Test correspondence first row
    assert all([PIC_lst[i][0] == paths_lst[i] for i in range(len(paths_lst))]), " unordered first row "


    # Test the expected values
    for idx, element in enumerate(PIC_lst):
        assert element[0] == paths_lst[idx] \
               and element[1] == idx \
               and element[2] == CONST.CLASSES.index(paths_lst[idx].split(os.sep)[-2]), \
            str(element) + " XXX doesn't have the expected PIC format for " + str(paths_lst[idx])


def test_replicate_PIC_list_by_number_of_replication():
    """Test for replicate_PIC_list_by_number_of_replication()"""

    # Call the function
    path_lst, idx_lst, label_lst  = get_test_PIC_elements()
    pic_train_paths = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    replications_simulated = [random.randint(1, 10) for _ in range(len(pic_train_paths))]
    pic_train_paths_replicated = replicate_PIC_list_by_number_of_replication(pic_train_paths, replications_simulated)

    # Test the length
    assert len(pic_train_paths_replicated) == sum(replications_simulated), \
        "There is not the same element as is expected: " \
        + str(len(pic_train_paths_replicated)) \
        + " instead of " + str(sum(replications_simulated))

    # Test each element is as time as is indicated
    offset = 0
    for idx, element in enumerate(pic_train_paths):
        element_replicate = pic_train_paths_replicated[offset:offset + replications_simulated[idx]]

        assert all([element_rep == element for element_rep in element_replicate]), \
            "There are elements which has not been replicated properly"

        offset += replications_simulated[idx]


def test_get_sample_by_class_from_PIC_list():
    """Test for get_sample_by_class_from_PIC_list()"""

    # Call the function
    path_lst, idx_lst, label_lst  = get_test_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    # Get the minimum number of element for all classes
    class_idx = [element[2] for element in pic]
    minimum_number_of_elements = min([class_idx.count(idx) for idx in range(len(CONST.CLASSES))])

    for number_of_samples_by_class in [int(x) for x in [minimum_number_of_elements / 2, minimum_number_of_elements / 3,
                                                        minimum_number_of_elements]]:
        subpic = get_sample_by_class_from_PIC_list(pic, number_of_samples_by_class)

        # Test the length
        assert len(subpic) == len(CONST.CLASSES) * number_of_samples_by_class, \
            "len(output) != len(CONST.CLASSES) * number_of_samples_by_class"

        # Test correspondence number of elements by class
        class_idx = [x[2] for x in subpic]
        assert set([class_idx.count(idx) for idx in range(len(CONST.CLASSES))]) == set([number_of_samples_by_class]), \
            "Wrong number of elements by class"


def test_parse_train():
    """Test for parse_train()"""

    # PIC training paths
    path_lst, idx_lst, label_lst  = get_train_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    # Checks for all paths
    NUM_TRAIN_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            input_data_tf, idx_src_new, idx_class_one_hot = parse_train(path, idx_src, idx_class)
            input_data_np = input_data_tf.numpy()
        except:
            print(path)

        # Do the tests
        assert input_data_tf.dtype == DATA_TYPE_INPUT, \
            "Input data type doesn't match"

        assert len(input_data_tf.shape) == 3, \
            "output is not a 3-D tensor"

        assert input_data_tf.shape[-1] == CONST.NUM_CHANNELS_INPUT and min(input_data_tf.shape[:-1]) > ANOMALOUS_SIZE, \
            "Input data values or shape doesn't match"

        assert all(np.min(input_data_np[:, :, channel])
                   != np.max(input_data_np[:, :, channel]) for channel in
                   range(CONST.NUM_CHANNELS_INPUT)), \
            "Input min and max value are the same. Maybe it's all zeros"

        assert idx_src_new == idx_src, \
            "idx_src have been modified"

        assert len(idx_class_one_hot.shape) == 1 and idx_class_one_hot.shape[0] == len(CONST.CLASSES), \
            "idx_class_one_hot doesn't have the expected shape"

        assert idx_class_one_hot.dtype == tf.dtypes.float32, \
            "idx_class_one_hot is not float32"

        assert idx_class_one_hot[idx_class] == 1 and tf.reduce_sum(idx_class_one_hot) == 1, \
            "idx_class_one_hot is not a correct one-hot vector for the class"

        if i % 500 == 0 or (i + 1) == len(path_lst):
            print("[test_parse_train()]:", i, "of", NUM_TRAIN_DATA)


def test_parse_val():
    """Test for parse_val()"""

    # PIC val paths
    path_lst, idx_lst, label_lst  = get_val_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)


    # Checks for all paths
    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            input_data_tf, idx_src_new, idx_class_one_hot = parse_val(path, idx_src, idx_class)
            input_data_np = input_data_tf.numpy()
        except:
            print(path)

        # Do the tests
        assert input_data_tf.dtype == DATA_TYPE_INPUT, \
            "Input data type doesn't match"

        assert len(input_data_tf.shape) == 3, \
            "output is not a 3-D tensor"

        assert input_data_tf.shape[-1] == CONST.NUM_CHANNELS_INPUT and min(input_data_tf.shape[:-1]) > ANOMALOUS_SIZE, \
            "Input data values or shape doesn't match"

        assert all(np.min(input_data_np[:, :, channel])
                   != np.max(input_data_np[:, :, channel]) for channel in
                   range(CONST.NUM_CHANNELS_INPUT)), \
            "Input min and max value are the same. Maybe it's all zeros"

        assert idx_src_new == idx_src, \
            "idx_src have been modified"

        assert len(idx_class_one_hot.shape) == 1 and idx_class_one_hot.shape[0] == len(CONST.CLASSES), \
            "idx_class_one_hot doesn't have the expected shape"

        assert idx_class_one_hot.dtype == tf.dtypes.float32, \
            "idx_class_one_hot is not float32"

        assert idx_class_one_hot[idx_class] == 1 and tf.reduce_sum(idx_class_one_hot) == 1, \
            "idx_class_one_hot is not a correct one-hot vector for the class"

        if i % 100 == 0 or (i + 1) == len(path_lst):
            print("[test_parse_val()]:", i, "of", NUM_DATA)


def test_parse_test():
    """Test for parse_test()"""

    # PIC test paths
    path_lst, idx_lst, label_lst  = get_test_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    # Checks for all paths
    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            input_data_tf, idx_src_new, idx_class_one_hot = parse_test(path, idx_src, idx_class)
            input_data_np = input_data_tf.numpy()
        except:
            print(path)


        # Do the tests
        assert input_data_tf.dtype == DATA_TYPE_INPUT, \
            "Input data type doesn't match" + path

        assert len(input_data_tf.shape) == 3, \
            "output is not a 3-D tensor"

        assert input_data_tf.shape[-1] == CONST.NUM_CHANNELS_INPUT and min(input_data_tf.shape[:-1]) > ANOMALOUS_SIZE, \
            "Input data values or shape doesn't match"

        assert all(np.min(input_data_np[:, :, channel])
                   != np.max(input_data_np[:, :, channel]) for channel in
                   range(CONST.NUM_CHANNELS_INPUT)), \
            "Input min and max value are the same. Maybe it's all zeros"

        assert idx_src_new == idx_src, \
            "idx_src have been modified"

        assert len(idx_class_one_hot.shape) == 1 and idx_class_one_hot.shape[0] == len(CONST.CLASSES), \
            "idx_class_one_hot doesn't have the expected shape"

        assert idx_class_one_hot.dtype == tf.dtypes.float32, \
            "idx_class_one_hot is not float32"

        assert idx_class_one_hot[idx_class] == 1 and tf.reduce_sum(idx_class_one_hot) == 1, \
            "idx_class_one_hot is not a correct one-hot vector for the class"

        if i % 100 == 0 or (i + 1) == len(path_lst):
            print("[test_parse_test()]:", i, "of", NUM_DATA)


def test_preprocess_train():
    """Tests for the function preprocess_train()"""
    # Get all the paths
    path_lst, idx_lst, label_lst = get_train_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            img_decode, idx_src_new, idx_class_one_hot = parse_train(path, idx_src, idx_class)
            img_transformed, idx_src_new_new, idx_class_one_hot_new = \
                preprocess_train(img_decode, idx_src_new, idx_class_one_hot)
        except:
            print(path)

        # Do the tests
        assert img_transformed.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        # print(img_decode.shape, img_transformed.shape)                  XXXXX
        #assert img_decode.shape == img_transformed.shape, \   XXXX
        #    "Input shape doesn't match"

        minimum = tf.reduce_min(img_transformed)
        maximum = tf.reduce_max(img_transformed)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img_transformed values are not in [0,1]"

        assert idx_src_new_new == idx_src and np.array_equal(idx_class_one_hot_new, idx_class_one_hot), \
            "idx_src and/or idx_class_one_hot have been modified"

        if i % 500 == 0 or (i + 1) == NUM_DATA:
            print("[test_preprocess_train()]:", i, "of", NUM_DATA)


def test_preprocess_val():
    """Tests for the function preprocess_val()"""
    # Get all the paths
    path_lst, idx_lst, label_lst = get_val_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            img_decode, idx_src_new, idx_class_one_hot = parse_val(path, idx_src, idx_class)
            img_transformed, idx_src_new_new, idx_class_one_hot_new = \
                preprocess_val(img_decode, idx_src_new, idx_class_one_hot)
        except:
            print(path)

        # Do the tests
        assert img_transformed.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        # print(img_decode.shape, img_transformed.shape)                  XXXXX
        #assert img_decode.shape == img_transformed.shape, \   XXXXX
        #    "Input shape doesn't match"

        minimum = tf.reduce_min(img_transformed)
        maximum = tf.reduce_max(img_transformed)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img_transformed values are not in [0,1]"

        assert idx_src_new_new == idx_src and np.array_equal(idx_class_one_hot_new, idx_class_one_hot), \
            "idx_src and/or idx_class_one_hot have been modified"

        if i % 100 == 0 or (i + 1) == NUM_DATA:
            print("[test_preprocess_val()]:", i, "of", NUM_DATA)


def test_preprocess_test():
    """Tests for the function preprocess_test()"""
    # Get all the paths
    path_lst, idx_lst, label_lst = get_test_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            img_decode, idx_src_new, idx_class_one_hot = parse_test(path, idx_src, idx_class)
            img_transformed, idx_src_new_new, idx_class_one_hot_new = \
                preprocess_test(img_decode, idx_src_new, idx_class_one_hot)
        except:
            print(path)

        # Do the tests
        assert img_transformed.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        # print(img_decode.shape, img_transformed.shape)                  XXXXX
        # assert img_decode.shape == img_transformed.shape, \
        #    "Input shape doesn't match"

        minimum = tf.reduce_min(img_transformed)
        maximum = tf.reduce_max(img_transformed)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img_transformed values are not in [0,1]"

        assert idx_src_new_new == idx_src and np.array_equal(idx_class_one_hot_new, idx_class_one_hot), \
            "idx_src and/or idx_class_one_hot have been modified"

        if i % 100 == 0 or (i + 1) == NUM_DATA:
            print("[test_preprocess_test()]:", i, "of", NUM_DATA)


def test_data_augmentation():
    """Tests for the function data_augmentation()"""

    probabilities = {BRIGHTNESS: 0.15, CONTRAST: 0.15, GAMMA: 0.15, HUE: 0.15,
                     SATURATION: 0.15, GAUSSIAN_NOISE: 0.15, BLUR: 0.15,
                     FLIP_VERTICAL: 0.15, FLIP_HORIZONTAL: 0.25,
                     ROTATION: 0.25, SALT_AND_PEPPER: 0.15,
                     SHEAR: 0.15}

    # Get all the paths
    path_lst, idx_lst, label_lst = get_val_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)


    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            img_decode, idx_src_new, idx_class_one_hot = parse_test(path, idx_src, idx_class)
            img_transformed, idx_src_new, idx_class_one_hot = \
                preprocess_test(img_decode, idx_src_new, idx_class_one_hot)
            img, idx_src_new, idx_class_one_hot_new = data_augmentation(img_transformed, idx_src_new, idx_class_one_hot,
                                                                        probabilities)
        except:
            print(path)

        assert img.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        assert img.shape == img_transformed.shape, \
            "Input shape doesn't match"

        minimum = tf.reduce_min(img)
        maximum = tf.reduce_max(img)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img values are not in [0,1]"

        assert idx_src_new == idx_src and np.array_equal(idx_class_one_hot_new, idx_class_one_hot), \
            "idx_src and/or idx_class_one_hot have been modified"

        if i % 100 == 0 or (i + 1) == NUM_DATA:
            print("[test_data_augmentation()]:", i, "of", NUM_DATA)

    # Test whether the function keep the original values
    probabilities = {BRIGHTNESS: 0.0, CONTRAST: 0.0, GAMMA: 0.0, HUE: 0.0,
                     SATURATION: 0.0, GAUSSIAN_NOISE: 0.0, BLUR: 0.0,
                     FLIP_VERTICAL: 0.0, FLIP_HORIZONTAL: 0.0,
                     ROTATION: 0.0, SALT_AND_PEPPER: 0.0,
                     SHEAR: 0.0}

    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            img_decode, idx_src_new, idx_class_one_hot = parse_test(path, idx_src, idx_class)
            img_transformed, idx_src_new, idx_class_one_hot = \
                preprocess_test(img_decode, idx_src_new, idx_class_one_hot)
            img, idx_src_new, idx_class_one_hot_new = data_augmentation(img_transformed, idx_src_new, idx_class_one_hot,
                                                                        probabilities)
        except:
            print(path)

        assert img.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        assert img.shape == img_transformed.shape, \
            "Input shape doesn't match"

        assert np.array_equal(img, img_transformed), \
            "Images are not equal"

        assert idx_src_new == idx_src and np.array_equal(idx_class_one_hot_new, idx_class_one_hot), \
            "idx_src and/or idx_class_one_hot have been modified"

        if i % 100 == 0 or (i + 1) == NUM_DATA:
            print("[test_data_augmentation()]:", i, "of", NUM_DATA)


def test_to_human_output():
    """Tests for the function to_human_output()"""

    # Get all the paths
    path_lst, idx_lst, label_lst = get_val_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)
    for i, element in enumerate(pic):

        # Extract each component
        path, idx_src, idx_class = element

        # Call the function
        try:
            img_decode, idx_src_new, idx_class_one_hot = parse_test(path, idx_src, idx_class)
            img_transformed, idx_src_new, idx_class_one_hot = \
                preprocess_test(img_decode, idx_src_new, idx_class_one_hot)
            img_human, class_human = to_human_output(img_transformed, idx_class_one_hot)
        except:
            print(path)

        assert tf.reduce_all(tf.equal(img_transformed, img_human)), \
            "Image has been modified"

        assert class_human == CONST.CLASSES[idx_class], \
            "class_human is not the expected name"

        if i % 100 == 0 or (i + 1) == NUM_DATA:
            print("[test_to_human_output()]:", i, "of", NUM_DATA)


def test_build_train_dataset():
    """Tests for the function build_train_dataset()"""

    probabilities = {BRIGHTNESS: 0.15, CONTRAST: 0.15, GAMMA: 0.15, HUE: 0.15,
                     SATURATION: 0.15, GAUSSIAN_NOISE: 0.15, BLUR: 0.15,
                     FLIP_VERTICAL: 0.15, FLIP_HORIZONTAL: 0.25,
                     ROTATION: 0.25, SALT_AND_PEPPER: 0.15,
                     SHEAR: 0.15}

    path_lst, idx_lst, label_lst = get_train_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)

    replications = [1 for _ in range(NUM_DATA)]

    # Get the minimum number of element for all classes
    class_idx = [element[2] for element in pic]
    minimum_number_of_elements = min([class_idx.count(idx) for idx in range(len(CONST.CLASSES))])
    num_examples_in_dataset = minimum_number_of_elements * len(CONST.CLASSES)

    # Call the function
    BATCH_SIZE = 1
    train_dataset = build_train_dataset(batch_size=BATCH_SIZE,
                                        num_batches_preloaded=5,
                                        num_parallel_calls=4,
                                        allow_repetitions=False,
                                        probabilities=probabilities,
                                        replications=replications,
                                        shuffle=True)

    # Test the dataset without repetitions
    expected_shape = [BATCH_SIZE, CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT]
    count = 0
    idx_used = []
    start_time = time.time()
    for img, idx_src, idx_class_one_hot in train_dataset:


        assert count < num_examples_in_dataset, \
            "Images generated are over the expected number: " + str(count) + "/" + str(num_examples_in_dataset)

        assert idx_src not in idx_used, \
            "Images are repited"
        idx_used.append(idx_src)

        assert idx_src >= 0 and idx_src < NUM_DATA, \
            "idx_src has an unexpected value: " + str(idx_src)

        assert idx_class_one_hot.ndim == 2 and idx_class_one_hot.shape[1] == len(CONST.CLASSES), \
            "idx_class_one_hot doesn't have the expected shape"

        assert idx_class_one_hot.dtype == tf.dtypes.float32, \
            "idx_class_one_hot is not float32"

        assert tf.reduce_sum(idx_class_one_hot) == 1 and idx_class_one_hot[0, pic[idx_src[0]][2]] == 1, \
            "idx_class_one_hot is not a correct one-hot vector for the class"

        assert img.ndim == 4, \
            "img is not a 4-D tensor"

        assert np.array_equal(img.shape, expected_shape), \
            "Image has not the expected shape: " \
            + str(img.shape) + " - " + str(expected_shape)

        assert img.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        minimum = tf.reduce_min(img)
        maximum = tf.reduce_max(img)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img values are not in [0,1]"

        if count % 500 == 0 or (count + 1) == NUM_DATA:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print("[test_build_train_dataset()]:", count, "of", np.ceil(NUM_DATA/BATCH_SIZE),"-- Elapsed time:", elapsed_time)

        count += 1

    """
    # Test the replication (NEED TO COMENT THE SAMPLE INSIDE THE FUNCTION)
    probabilities = {BRIGHTNESS: 0.0, CONTRAST: 0.0, GAMMA: 0.0, HUE: 0.0,
                    SATURATION: 0.0, GAUSSIAN_NOISE: 0.0, BLUR: 0.15,
                    FLIP_VERTICAL: 0.0, FLIP_HORIZONTAL: 0.0,
                    ROTATION: 0.0, SALT_AND_PEPPER: 0.0, SHIFT: 0.0,
                    SHEAR: 0.0}
    replications = [2 for _ in range(NUM_DATA)]

    # Call the function
    BATCH_SIZE = 1
    train_dataset = build_train_dataset(batch_size=BATCH_SIZE, 
                                        num_batches_preloaded=5, 
                                        num_parallel_calls=4,
                                        allow_repetitions=False, 
                                        probabilities=probabilities,
                                        replications=replications,
                                        shuffle=False)
    count = 0
    for _, _, _ in train_dataset: 
        count += 1
        if count % 100 == 0 or (count + 1) == NUM_DATA:
            print("[test_build_train_dataset()]:", count, "of", NUM_DATA)

    assert count == NUM_DATA*2, \
        "Images has not been replicated"
    """


def test_build_val_dataset():
    """Tests for the function build_val_dataset()"""

    path_lst, idx_lst, label_lst = get_val_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)

    # Get the minimum number of element for all classes

    # Call the function
    BATCH_SIZE = 1
    val_dataset = build_val_dataset(batch_size=BATCH_SIZE, 
                                        num_batches_preloaded=5, 
                                        num_parallel_calls=4)

    
    # Test the dataset without repetitions
    expected_shape = [BATCH_SIZE, CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT]
    count = 0
    idx_used = []
    for img, idx_src, idx_class_one_hot in val_dataset:       

        assert count < NUM_DATA, \
            "Images generated are over the expected number: " + str(count) + "/" + str(NUM_DATA)

        assert idx_src not in idx_used, \
            "Images are repited"
        idx_used.append(idx_src)

        assert idx_src >= 0 and idx_src < NUM_DATA, \
            "idx_src has an unexpected value: " + str(idx_src)

        assert idx_class_one_hot.ndim == 2 and idx_class_one_hot.shape[1] == len(CONST.CLASSES), \
            "idx_class_one_hot doesn't have the expected shape"
        
        assert idx_class_one_hot.dtype == tf.dtypes.float32, \
            "idx_class_one_hot is not float32"

        assert tf.reduce_sum(idx_class_one_hot) == 1 and idx_class_one_hot[0, pic[idx_src[0]][2]] == 1, \
            "idx_class_one_hot is not a correct one-hot vector for the class"

        assert img.ndim == 4, \
            "img is not a 4-D tensor"

        assert np.array_equal(img.shape, expected_shape), \
            "Image has not the expected shape: " \
            + str(img.shape) + " - " + str(expected_shape)

        assert img.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        minimum = tf.reduce_min(img)
        maximum = tf.reduce_max(img)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img values are not in [0,1]"

        if count % 100 == 0 or (count + 1) == NUM_DATA:
            print("[test_build_val_dataset()]:", count, "of", NUM_DATA)

        count += 1

    assert count == NUM_DATA, \
        "There are not the expected number of elements: " + str(count) + "/" + str(NUM_DATA)


def test_build_test_dataset():
    """Tests for the function build_test_dataset()"""

    path_lst, idx_lst, label_lst = get_test_PIC_elements()
    pic = pathlabelidx_lists_to_PIC_format(path_lst, label_lst, idx_lst)

    NUM_DATA = len(path_lst)

    # Get the minimum number of element for all classes

    # Call the function
    BATCH_SIZE = 1
    test_dataset = build_test_dataset(batch_size=BATCH_SIZE, 
                                        num_batches_preloaded=5, 
                                        num_parallel_calls=4)

    
    # Test the dataset without repetitions
    expected_shape = [BATCH_SIZE, CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT]
    count = 0
    idx_used = []
    for img, idx_src, idx_class_one_hot in test_dataset:       

        assert count < NUM_DATA, \
            "Images generated are over the expected number: " + str(count) + "/" + str(NUM_DATA)

        assert idx_src not in idx_used, \
            "Images are repited"
        idx_used.append(idx_src)

        assert idx_src >= 0 and idx_src < NUM_DATA, \
            "idx_src has an unexpected value: " + str(idx_src)

        assert idx_class_one_hot.ndim == 2 and idx_class_one_hot.shape[1] == len(CONST.CLASSES), \
            "idx_class_one_hot doesn't have the expected shape"
        
        assert idx_class_one_hot.dtype == tf.dtypes.float32, \
            "idx_class_one_hot is not float32"

        assert tf.reduce_sum(idx_class_one_hot) == 1 and idx_class_one_hot[0, pic[idx_src[0]][2]] == 1, \
            "idx_class_one_hot is not a correct one-hot vector for the class"

        assert img.ndim == 4, \
            "img is not a 4-D tensor"

        assert np.array_equal(img.shape, expected_shape), \
            "Image has not the expected shape: " \
            + str(img.shape) + " - " + str(expected_shape)

        assert img.dtype == tf.dtypes.float32, \
            "Input data type doesn't match"

        minimum = tf.reduce_min(img)
        maximum = tf.reduce_max(img)
        assert minimum >= 0.0 and maximum <= 1.0 and minimum != maximum, \
            "img values are not in [0,1]"

        if count % 100 == 0 or (count + 1) == NUM_DATA:
            print("[test_build_test_dataset()]:", count, "of", NUM_DATA)

        count += 1

    assert count == NUM_DATA, \
        "There are not the expected number of elements: " + str(count) + "/" + str(NUM_DATA)


def do_tests():
    """Launch all test avaiable in this module"""

    test_get_train_PIC_elements()
    
    test_get_test_PIC_elements()

    test_get_val_PIC_elements()
    
    test_pathlabelidx_lists_to_PIC_format()

    test_replicate_PIC_list_by_number_of_replication()
    
    test_get_sample_by_class_from_PIC_list()

    test_parse_train()
 
    test_parse_val()
   
    test_parse_test()

    test_preprocess_train()

    test_preprocess_val()

    test_preprocess_test()
    
    test_to_human_output()

    test_data_augmentation()

    test_build_train_dataset()

    test_build_val_dataset()

    test_build_test_dataset()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Only launch all tests
    do_tests()


# -----------------------------------------------------------------------------
# Information
# -----------------------------------------------------------------------------
"""
    - For each CSV file there are only two columns: Path and Class name
        - The first row is ignored
        - The class name have be in the variable CONST.CLASSES
"""



