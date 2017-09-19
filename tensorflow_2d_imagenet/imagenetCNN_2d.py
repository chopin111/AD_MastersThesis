import gc
import tensorflow as tf
import numpy as np

import os
import io

import random
import time
from datetime import date

import matplotlib.pyplot as plt

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
FLAGS.width = 3
FLAGS.height = 224
FLAGS.depth = 224 # 3

FLAGS.iterations = 20000 # 3

FLAGS.batch_size = 50

FLAGS.train_update_count = 10
FLAGS.test_update_count = 100
FLAGS.checkpoint = 1000

TODAY_DATE = str(date.today())
try:
	os.mkdir('./' + TODAY_DATE)
except:
	pass

FLAGS.save_path = './' + TODAY_DATE + '/model.ckpt'
FLAGS.telemetry_path = './' + TODAY_DATE + '/telemetry.csv'

FLAGS.logs_path = './' + TODAY_DATE + '/logs'

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
	image = tf.reshape(image, [FLAGS.width,FLAGS.depth,FLAGS.height])
	image = tf.transpose(image, perm=[1, 2, 0]) # move channels to back
	image = tf.reshape(image, tf.stack([FLAGS.width*FLAGS.height*FLAGS.depth]))

	label=tf.stack(tf.one_hot(label-1, nLabel))
	return label, image



train_files = ["/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00000-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00001-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00002-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00003-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00004-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00005-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00006-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00007-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00008-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00009-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00010-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00011-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00012-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00013-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00014-of-00016",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/train-00015-of-00016"]

validation_files = ["/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00000-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00001-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00002-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00003-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00004-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00005-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00006-of-00008",
"/media/piotr/CE58632058630695/data-tf-2d-distorted/validation-00007-of-00008"]

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

with tf.name_scope("input"):
	x = tf.placeholder(tf.float32, shape=[None, FLAGS.width*FLAGS.height*FLAGS.depth], name="x-input") # [None, 28*28]
	y_ = tf.placeholder(tf.float32, shape=[None, nLabel], name="y-input")  # [None, 10]


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1,FLAGS.depth,FLAGS.height,FLAGS.width]) # as width is a 'channel', lets move it to back

net = tf.layers.conv2d(x_image,
	96, [11, 11], 4,
	#kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
	padding="SAME",
	activation=tf.nn.relu,
	name="conv2d_0")
#net = tf.layers.batch_normalization(
#	inputs=net,
#	#axis=-1, momentum=0.999,
#	#epsilon=0.001, center=True, scale=True,
#	training=is_training,
#	name="batch_normalization_0")
net = tf.layers.max_pooling2d(net, [3, 3], 2, name='max_pool_2d_0')
#net = relu_0

net = tf.layers.conv2d(net,
	256, [5, 5], 4,
	kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
	padding="SAME",
	activation=tf.nn.relu,
	name="conv2d_1")
net = tf.layers.batch_normalization(
	inputs=net,
	#momentum=0.999, epsilon=0.001,
	#center=True, scale=True,
	training=is_training,
	name="batch_normalization_1")
net = tf.layers.max_pooling2d(net, [3, 3], 2, name='max_pool_2d_1')

net = tf.layers.conv2d(net,
	384, [3, 3], 4,
	#kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
	padding="SAME",
	activation=tf.nn.relu,
	name="conv2d_2")
net = tf.layers.conv2d(net,
	384, [3, 3], 4,
#	use_bias=True,
#	bias_initializer=tf.zeros_initializer(),
	#kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
	padding="SAME",
	activation=tf.nn.relu,
	name="conv2d_3")
net = tf.layers.conv2d(net,
	384, [3, 3], 4,
#	use_bias=True,
#	bias_initializer=tf.zeros_initializer(),
	#kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
	activation=tf.nn.relu,
	padding="SAME",
	name="conv2d_4")

net = tf.contrib.layers.flatten(net)
net = tf.contrib.layers.fully_connected(net, 2048)
net = tf.nn.dropout(net, keep_prob)
net = tf.contrib.layers.fully_connected(net, 2048)
net = tf.nn.dropout(net, keep_prob)
y_conv = tf.contrib.layers.fully_connected(net, nLabel)


extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	with tf.name_scope('train'):
		#train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
		#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 1e-4

		optimizer = tf.train.AdamOptimizer(1e-7)
		gradients = optimizer.compute_gradients(cross_entropy)
		capped_gradients = [(tf.clip_by_value(grad, 1e-10, 1.0), var) for grad, var in gradients]
		train_step = optimizer.apply_gradients(capped_gradients)

		#optimizer = tf.train.AdamOptimizer(1e-3)
		#gradients, variables = zip(*optimizer.compute_gradients(cross_entropy))
		#gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
		#train_step = optimizer.apply_gradients(zip(gradients, variables))
	with tf.name_scope('Accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('cost', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

for grad, var in gradients:
	tf.summary.histogram(var.name + '/gradient', grad)

summary_op = tf.summary.merge_all()

train_cost_summary = tf.summary.scalar('train_cost', cross_entropy)
test_cost_summary = tf.summary.scalar('test_cost', cross_entropy)
train_accuracy_summary = tf.summary.scalar('train_accuracy', accuracy)
test_accuracy_summary = tf.summary.scalar('test_accuracy', accuracy)


sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

writer = tf.summary.FileWriter(FLAGS.logs_path, graph=tf.get_default_graph())

train_accuracy = 0
test_accuracy = 0

saver = tf.train.Saver()

# Include keep_prob in feed_dict to control dropout rate.
for i in range(FLAGS.iterations):
	batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

	if i%FLAGS.train_update_count == 0:
		train_accuracy, train_entropy, train_cost_summ, train_accuracy_summ = sess.run([accuracy, cross_entropy, train_cost_summary, train_accuracy_summary], {x:batch_xs, y_: batch_ys, keep_prob: 1.0, is_training: False})
		writer.add_summary(train_cost_summ, i)
		writer.add_summary(train_accuracy_summ, i)
	if i%FLAGS.test_update_count == 0:
		vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
		test_accuracy, test_entropy, test_cost_summ, test_accuracy_summ = sess.run([accuracy, cross_entropy, test_cost_summary, test_accuracy_summary], {x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, is_training: False})
		writer.add_summary(test_cost_summ, i)
		writer.add_summary(test_accuracy_summ, i)

	print("step %d, train acc: %g, test acc:%g                                       " % (i, train_accuracy, test_accuracy), end='\r')
	_, summary = sess.run([train_step, summary_op], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75, is_training: True})
	writer.add_summary(summary, i)

	if i%FLAGS.checkpoint == 0:
		saver.save(sess, FLAGS.save_path, i)

# Evaulate our accuracy on the test data
vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
print("test accuracy %g"%accuracy.eval(feed_dict={x: vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, is_training: False}))

save_path = saver.save(sess, FLAGS.save_path)
