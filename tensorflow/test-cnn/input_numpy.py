# NOTE: heavily based on https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.learn.python.learn.datasets import mnist

import argparse
import sys

import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import pandas
import skimage.io
import skimage.transform
import xml.etree.ElementTree as ET

import tensorflow as tf

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
NUM_CLASSES = 31
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

    fullpath = IMAGES_PATH
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

    return images_filenames, labels_filenames

class BrainData(object):
    def __init__(self, images_filenames, labels):
        self.num_examples = 0
        self.curr_batch = 0
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_first_epoch = True
        self.load_data(images_filenames, labels)

    def load_data(self, images_filenames, labels):
        self.num_examples = len(images_filenames)

        self.total_images = np.ndarray((self.num_examples, RESULT_IMAGE_SIZE*RESULT_IMAGE_SIZE))
        self.total_labels = np.ndarray((self.num_examples, NUM_CLASSES))

        i = 0

        for image_filename in images_filenames:
            curr_image = imread(image_filename)

            graysmall = imresize(curr_image, [RESULT_IMAGE_SIZE, RESULT_IMAGE_SIZE])/255.
            grayvec = np.reshape(graysmall, (1, -1))

            self.total_images[i, :] = grayvec
            self.total_labels[i, :] = np.zeros(NUM_CLASSES)
            self.total_labels[i, labels[i]] = 1
            i = i + 1
        print (self.total_labels)

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples or self.is_first_epoch == True:
          self.is_first_epoch = False
          self.epochs_completed += 1

          perm = np.arange(self.num_examples)
          np.random.shuffle(perm)
          self.total_images = self.total_images[perm]
          self.total_labels = self.total_labels[perm]

          start = 0
          self.index_in_epoch = batch_size
          assert batch_size <= self.num_examples
        end = self.index_in_epoch
        print ("%d %d" % (start, end))
        return self.total_images[start:end], self.total_labels[start:end]

def read_inputs(num_epochs, batch_size, filepath):
    images_filenames, labels_filenames = read_labeled_images_list(filepath)

    return BrainData(images_filenames, labels_filenames)


if __name__ == '__main__':
    read_inputs(10, 10)
