# Start TensorFlow InteractiveSession
import gc
import tensorflow as tf
import numpy as np

import os

import random
import time
from datetime import date

# Path hack.
import sys
sys.path.insert(0, os.path.abspath('..'))

from learnutils.visualization import DashboardVis, Telemetry

######################################
######################################
######################################
######      DATA LOADING        ######
######################################
######################################
######################################

nLabel = 5

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 45
FLAGS.height = 128
FLAGS.depth = 80 # 3

FLAGS.iterations = 60000 # 3

FLAGS.batch_size = 10

FLAGS.train_update_count = 10
FLAGS.test_update_count = 100

TODAY_DATE = str(date.today())
try:
	os.mkdir('./' + TODAY_DATE)
except:
	pass

FLAGS.save_path = './' + TODAY_DATE + '/model.ckpt'
FLAGS.telemetry_path = './' + TODAY_DATE + '/telemetry.csv'

# Function to tell TensorFlow how to read a single image from input file
def getImage(filenames):
	# convert filenames to a queue for an input pipeline.
	filenameQ = tf.train.string_input_producer(filenames,num_epochs=None)

	# object to read records
	recordReader = tf.TFRecordReader()

	# read the full set of features for a single example
	key, fullExample = recordReader.read(filenameQ)

	# parse the full example into its' component features.
	features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

	label = features['image/class/label']
	image_buffer = features['image/encoded']

	image = tf.decode_raw(image_buffer, tf.float32)
	image = tf.reshape(image, tf.stack([FLAGS.width*FLAGS.height*FLAGS.depth]))

	label=tf.stack(tf.one_hot(label-1, nLabel))
	return label, image



train_files = ["/media/piotr/CE58632058630695/data-tf-5/train-00000-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00001-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00002-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00003-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00004-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00005-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00006-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00007-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00008-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00009-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00010-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00011-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00012-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00013-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00014-of-00016",
"/media/piotr/CE58632058630695/data-tf-5/train-00015-of-00016"]

validation_files = ["/media/piotr/CE58632058630695/data-tf-5/validation-00000-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00001-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00002-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00003-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00004-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00005-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00006-of-00008",
"/media/piotr/CE58632058630695/data-tf-5/validation-00007-of-00008"]

label, image = getImage(train_files)
vlabel, vimage = getImage(validation_files)

imageBatch, labelBatch = tf.train.shuffle_batch(
	[image, label], batch_size=FLAGS.batch_size,
	capacity=400,
	min_after_dequeue=100)

vimageBatch, vlabelBatch = tf.train.shuffle_batch(
	[vimage, vlabel], batch_size=FLAGS.batch_size,
	capacity=400,
	min_after_dequeue=100)

######################################
######################################
######################################
######    END DATA LOADING      ######
######################################
######################################
######################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
# Placeholders (MNIST image:28x28pixels=784, label=10)
#with tf.device('/gpu:0'):
x = tf.placeholder(tf.float32, shape=[None, FLAGS.width*FLAGS.height*FLAGS.depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]
#withend

## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_1(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
	return tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1], strides=[1, 3, 3, 3, 1], padding='SAME')

def max_pool_2(x):
	return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')

## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,FLAGS.width,FLAGS.height,FLAGS.depth,1]) # [-1,28,28,1]
print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print("shape: %s" % h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
h_pool1 = max_pool_1(h_conv1)  # apply max-pool
print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32


## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64]) # [64]

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
h_pool2 = max_pool_2(h_conv2)  # apply max-pool
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64

## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([4*11*7*64, 512])  # [7*7*64, 512]
b_fc1 = bias_variable([512]) # [512]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*11*7*64])  # -> output image: [-1, 7*7*64] = 3136
print(h_pool2_flat.get_shape)  # (?, 2621440)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
print(h_fc1.get_shape) # (?, 512)  # -> output: 512
gc.collect()

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  # -> output: 512

## Readout Layer
W_fc2 = weight_variable([512, nLabel]) # [512, 31]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

vis = DashboardVis(FLAGS.width, FLAGS.height)
telemetry = Telemetry(FLAGS.telemetry_path)
telemetry.create()

# Include keep_prob in feed_dict to control dropout rate.
for i in range(FLAGS.iterations):
	start_time = time.time()
	print("step %d"%(i))
	batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

	"""if i%1 == 0:
		run_metadata = tf.RunMetadata()
		sess.run([accuracy, cross_entropy], {x:batch_xs, y_: batch_ys, keep_prob: 1.0}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
		from tensorflow.python.client import timeline
		trace = timeline.Timeline(step_stats=run_metadata.step_stats)
		trace_file = open('timeline.ctf.json', 'w')
		trace_file.write(trace.generate_chrome_trace_format())"""


	if i%FLAGS.train_update_count == 0:
		train_accuracy, train_entropy = sess.run([accuracy, cross_entropy], {x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		vis.addAccuracy(i, train_accuracy, 'train')
		vis.addCrossEntropyLoss(i, train_entropy, 'train')
		telemetry.addAccuracy(i, train_accuracy, 'train')
		telemetry.addCrossEntropyLoss(i, train_entropy, 'train')
		print("step %d, training accuracy %g"%(i, train_accuracy))
	if i%FLAGS.test_update_count == 0:
		vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
		test_accuracy, test_entropy = sess.run([accuracy, cross_entropy], {x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})

		vis.addAccuracy(i, test_accuracy, 'test')
		vis.addCrossEntropyLoss(i, test_entropy, 'test')

		telemetry.addAccuracy(i, test_accuracy, 'test')
		telemetry.addCrossEntropyLoss(i, test_entropy, 'test')

	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})
	if True == False:
		vis.updateSliceInd((i%10)/10)
		if i%100 == 0:
			# draw random image from batch_xs
			singleMri = batch_xs[0, :]
			reshaped = np.reshape(singleMri, (FLAGS.width, FLAGS.height, FLAGS.depth))

			vis.updateImage(reshaped)

		vis.update()
		vis.draw()
	gc.collect()
	duration = time.time() - start_time
	print("%.2f" % duration)
	telemetry.addStepExecutionTime(i, duration)

# Evaulate our accuracy on the test data
vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
print("test accuracy %g"%accuracy.eval(feed_dict={x: vbatch_xs, y_: vbatch_ys, keep_prob: 1.0}))

saver = tf.train.Saver()
save_path = saver.save(sess, FLAGS.save_path)
