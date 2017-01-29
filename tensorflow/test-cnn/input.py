# NOTE: heavily based on https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/

import os

import tensorflow as tf

import xml.etree.ElementTree as ET

# Files info
METADATA_PATH_SUFFIX = "./metadata.xml"
VALID_EXTENSIONS = [".jpg", ".gif", ".png", ".tga", ".jpeg"]

# Images to be resized to IMAGE_SIZE x IMAGE_SIZE
# 28 - mnist standard size
RESULT_IMAGE_SIZE = 28
RESULT_IMAGE_DEPTH = 1

# Source image sizes
SOURCE_IMAGE_WIDTH = 256
SOURCE_IMAGE_HEIGHT = 256
# only a grey channel is available
SOURCE_IMAGE_DEPTH = 1

# Global constants for input data
NUM_CLASSES = 30
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Should be converted from 2d to 1d
SHOULD_FLATTEN = True

#TODO: for now I assume that every file is jpg
def read_images_from_disk(input_queue):
    """ Read one file in filename url and adds label passed as variable.
    Return result object with fields:
        label -> tf.int32 with class number
        width, height, depth - ints

    """
    label = input_queue[1]
    raw_data = tf.read_file(input_queue[0])
    decoded = tf.image.decode_jpeg(raw_data, channels=SOURCE_IMAGE_DEPTH)
    decoded.set_shape([SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_DEPTH])

    resized = tf.image.resize_images(decoded, [RESULT_IMAGE_SIZE, RESULT_IMAGE_SIZE])

    if SHOULD_FLATTEN:
        resized = tf.reshape(resized, [-1])
        #GREYSCALE TO FLOAT
        tf.cast(resized, tf.float32)
        x = tf.constant(255.0, shape=[RESULT_IMAGE_SIZE*RESULT_IMAGE_SIZE])
        divided = tf.divide(resized, x)

        return divided, label

    return resized, label

def read_labeled_images_list(filepath):
    """ Read filename with metadata and load image data as well as labels """
    labels = {}

    METADATA_PATH = filepath + "/" + METADATA_PATH_SUFFIX
    IMAGES_PATH = filepath

    tree = ET.parse(METADATA_PATH)
    metadata = tree.getroot()
    for image in metadata:
        name = image.attrib['name']
        mmse = image.find('MMSE')
        mmseValue = mmse.text
        labels[name] = int(float(mmseValue)) #TODO: czy moga byc inty?

    cwd = os.getcwd()

    image_count = 0

    images_filenames = []
    labels_filenames = []

    fullpath = cwd
    files_list = os.listdir(fullpath)
    for f in files_list:
        name, ext = os.path.splitext(f)
        if ext.lower() not in VALID_EXTENSIONS:
            continue
        if name not in labels:
            continue
        image_filename = fullpath + "/" + f
        image_label = labels[name]
        images_filenames.append(image_filename)
        labels_filenames.append(image_label)
        print image_filename

    return images_filenames, labels_filenames

def read_inputs(num_epochs, batch_size, filepath):
    images_filenames, labels_filenames = read_labeled_images_list(filepath)

    images = tf.convert_to_tensor(images_filenames, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_filenames, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=num_epochs,
                                            shuffle=True)

    image, label = read_images_from_disk(input_queue)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size)

    return image_batch, label_batch

if __name__ == '__main__':
    read_inputs(10, 10)
