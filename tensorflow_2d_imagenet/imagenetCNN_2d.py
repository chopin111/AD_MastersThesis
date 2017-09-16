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

nLabel = 2

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 224
FLAGS.height = 224
FLAGS.depth = 3 # 3

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
			#'image/depth': tf.FixedLenFeature([], tf.int64),
			'image/class/label': tf.FixedLenFeature([],tf.int64),
			'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
			'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
			'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
		})

	label = features['image/class/label']
	image_buffer = features['image/encoded']

	image = tf.decode_raw(image_buffer, tf.float32)
	#image = tf.reshape(image, tf.stack([FLAGS.width*FLAGS.height*FLAGS.depth]))

	with tf.name_scope('decode_jpeg',[image_buffer], None):
		# decode
		image = tf.image.decode_jpeg(image_buffer, channels=3)

		# and convert to single precision data type
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)

	image = tf.reshape(image, tf.stack([FLAGS.width*FLAGS.height*FLAGS.depth]))

	label=tf.stack(tf.one_hot(label-1, nLabel))
	return label, image


train_files = ["data/train-00000-of-00001"]

validation_files = ["data/validation-00000-of-00001"]

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


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1,FLAGS.width,FLAGS.height,FLAGS.depth]) # [-1,45,128,80,1]

net = tf.layers.conv2d(x_image,
	96, [11, 11], 4,
	use_bias=True,
	bias_initializer=tf.zeros_initializer(),
	kernel_initializer=tf.truncated_normal_initializer(),
	padding="SAME",
	name="conv2d_0")
#batch_normalization_0 = conv2d_0
net = tf.layers.batch_normalization(
	inputs=net, axis=-1, momentum=0.999,
	epsilon=0.001, center=True, scale=True,
	training=is_training,
	name="batch_normalization_0")
net = tf.layers.max_pooling2d(net, [3, 3], 2, name='max_pool_2d_0')
net = tf.nn.relu(net, name="relu_0")
#net = relu_0

net = tf.layers.conv2d(net,
	256, [5, 5], 4,
	use_bias=True,
	bias_initializer=tf.zeros_initializer(),
	kernel_initializer=tf.truncated_normal_initializer(),
	padding="SAME",
	name="conv2d_1")
net = tf.layers.batch_normalization(
	inputs=net, momentum=0.999, epsilon=0.001,
	center=True, scale=True,
	training=is_training,
	name="batch_normalization_1")
net = tf.layers.max_pooling2d(net, [3, 3], 2, name='max_pool_2d_1')
net = tf.nn.relu(net, name="relu_1")

net = tf.layers.conv2d(net,
	384, [3, 3], 4,
	use_bias=True,
	bias_initializer=tf.zeros_initializer(),
	kernel_initializer=tf.truncated_normal_initializer(),
	padding="SAME",
	name="conv2d_2")
#net = tf.layers.conv2d(net,
#	384, [3, 3], 4,
#	use_bias=True,
#	bias_initializer=tf.zeros_initializer(),
#	kernel_initializer=tf.truncated_normal_initializer(),
#	padding="SAME",
#	name="conv2d_3")
#net = tf.layers.conv2d(net,
#	384, [3, 3], 4,
#	use_bias=True,
#	bias_initializer=tf.zeros_initializer(),
#	kernel_initializer=tf.truncated_normal_initializer(),
#	padding="SAME",
#	name="conv2d_4")

net = tf.contrib.layers.flatten(net)
net = tf.contrib.layers.fully_connected(net, 2048)
net = tf.nn.dropout(net, keep_prob)
net = tf.contrib.layers.fully_connected(net, 2048)
net = tf.nn.dropout(net, keep_prob)
y_conv = tf.contrib.layers.fully_connected(net, nLabel)


extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

# Include keep_prob in feed_dict to control dropout rate.
for i in range(FLAGS.iterations):
	start_time = time.time()
	print("step %d"%(i))
	batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

	if i%FLAGS.train_update_count == 0:
		train_accuracy, train_entropy = sess.run([accuracy, cross_entropy], {x:batch_xs, y_: batch_ys, keep_prob: 1.0, is_training: False})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	if i%FLAGS.test_update_count == 0:
		vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
		test_accuracy, test_entropy = sess.run([accuracy, cross_entropy], {x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, is_training: False})
		print("step %d, test accuracy %g"%(i, test_accuracy))

	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5, is_training: True})


	gc.collect()
	duration = time.time() - start_time
	print("%.2f" % duration)

# Evaulate our accuracy on the test data
vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
print("test accuracy %g"%accuracy.eval(feed_dict={x: vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, is_training: False}))

saver = tf.train.Saver()
save_path = saver.save(sess, FLAGS.save_path)
