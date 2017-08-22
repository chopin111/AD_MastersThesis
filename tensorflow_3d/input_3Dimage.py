import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib
import xml.etree.ElementTree

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 128
FLAGS.height = 128
FLAGS.depth = 40 # 3
batch_index = 0
filenames = []

# user selection
FLAGS.data_dir = '... do uzupelnienia sciezka do plikow nii'

def get_filenames(data_set):
    global filenames
#    labels = []

    e = xml.etree.ElementTree.parse(FLAGS.data_dir + '/metadata.xml').getroot()
    i = 0
    
    for image in e.findall('image'):
        filename = image.get('name') + '.nii'
        for mmse in image.findall('MMSE'):
            mmse = mmse.text
        filenames.append(['/' + filename, i, mmse])
        i += 1
        if i == 100:
            break

    print(filenames)
    random.shuffle(filenames)


def get_data_MRI(sess, data_set, batch_size):
    global batch_index, filenames

    if len(filenames) == 0: 
        get_filenames(data_set) 
        
    max = len(filenames)

    begin = batch_index
    end = batch_index + batch_size

    if end >= max:
        end = max
        batch_index = 0

    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, 31)) # zero-filled list for 'one hot encoding'
    index = 0

    for i in range(begin, end):
        imagePath = FLAGS.data_dir + '/' + data_set + '/' + filenames[i][0]
        FA_org = nib.load(imagePath)
        FA_data = FA_org.get_data()  # 256x256x40; numpy.ndarray
        depth = FA_data.shape[2]
        FA_data = np.delete(FA_data, range(FLAGS.depth, depth), axis = 2)
        tensor = tf.convert_to_tensor(FA_data, dtype=tf.float32)
        
        # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
        resized_image = tf.image.resize_images(images=tensor, size=(FLAGS.width,FLAGS.height), method=1)

        image = sess.run(resized_image)  # (256,256,40)
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        y_data[index][int(float(filenames[i][2]))] = 1  # assign mmse to corresponding column (one hot encoding)
        index += 1
        print('Done img ' + repr(i) + ' out of ' + repr(max))

    batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth)

    return x_data_, y_data