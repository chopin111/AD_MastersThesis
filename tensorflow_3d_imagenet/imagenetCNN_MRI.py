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


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1,FLAGS.width,FLAGS.height,FLAGS.depth,1]) # [-1,45,128,80,1]

conv2d_0 = tf.layers.conv3d(x_image,
	96, [FLAGS.width, 11, 11], [4, 4, 4],
	use_bias=True,
	bias_initializer=tf.zeros_initializer(),
	kernel_initializer=tf.truncated_normal_initializer(),
	padding="SAME",
	name="conv2d_0")
#batch_normalization_0 = conv2d_0
batch_normalization_0 = tf.layers.batch_normalization(
	inputs=conv2d_0, axis=-1, momentum=0.999,
	epsilon=0.001, center=True, scale=True,
	training=is_training,
	name="batch_normalization_0")
max_pool_3d_0 = tf.layers.max_pooling3d(batch_normalization_0, [3, 3, 3], 2, name='max_pool_3d_0')
relu_0 = tf.nn.relu(max_pool_3d_0, name="relu_0")
#net = relu_0

conv2d_1 = tf.layers.conv3d(relu_0,
	256, [5, 5, 48], [4, 4, 4],
	use_bias=True,
	bias_initializer=tf.zeros_initializer(),
	kernel_initializer=tf.truncated_normal_initializer(),
	padding="SAME",
	name="conv2d_1")
batch_normalization_1 = tf.layers.batch_normalization(
	inputs=conv2d_1, momentum=0.999, epsilon=0.001,
	center=True, scale=True,
	training=is_training,
	name="batch_normalization_1")
max_pool_3d_1 = tf.layers.max_pooling3d(batch_normalization_1, [3, 3, 3], 2, padding="SAME", name='max_pool_3d_1')
net = tf.nn.relu(max_pool_3d_1, name="relu_1")

net = tf.layers.conv3d(net,
	384, [3, 3, 256], [4, 4, 4],
	use_bias=True,
	bias_initializer=tf.zeros_initializer(),
	kernel_initializer=tf.truncated_normal_initializer(),
	padding="SAME",
	name="conv2d_2")
#net = tf.layers.conv3d(net,
#	384, [3, 3, 256], [4, 4, 4],
#	use_bias=True,
#	bias_initializer=tf.zeros_initializer(),
#	kernel_initializer=tf.truncated_normal_initializer(),
#	padding="SAME",
#	name="conv2d_3")
#net = tf.layers.conv3d(net,
#	384, [3, 3, 256], [4, 4, 4],
#	use_bias=True,
#	bias_initializer=tf.zeros_initializer(),
#	kernel_initializer=tf.truncated_normal_initializer(),
#	padding="SAME",
#	name="conv2d_4")

flattened_net = tf.contrib.layers.flatten(net)
fully_connected_0 = tf.contrib.layers.fully_connected(flattened_net, 4096)
dropout_0 = tf.nn.dropout(fully_connected_0, keep_prob)
fully_connected_1 = tf.contrib.layers.fully_connected(dropout_0, 4096)
dropout_1 = tf.nn.dropout(fully_connected_1, keep_prob)
y_conv = tf.contrib.layers.fully_connected(dropout_1, nLabel)


extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
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

	if i%FLAGS.train_update_count == 0:
		train_accuracy, train_entropy = sess.run([accuracy, cross_entropy], {x:batch_xs, y_: batch_ys, keep_prob: 1.0, is_training: False})
		vis.addAccuracy(i, train_accuracy, 'train')
		vis.addCrossEntropyLoss(i, train_entropy, 'train')
		telemetry.addAccuracy(i, train_accuracy, 'train')
		telemetry.addCrossEntropyLoss(i, train_entropy, 'train')
		print("step %d, training accuracy %g"%(i, train_accuracy))
	if i%FLAGS.test_update_count == 0:
		vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
		test_accuracy, test_entropy = sess.run([accuracy, cross_entropy], {x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, is_training: False})

		vis.addAccuracy(i, test_accuracy, 'test')
		vis.addCrossEntropyLoss(i, test_entropy, 'test')

		telemetry.addAccuracy(i, test_accuracy, 'test')
		telemetry.addCrossEntropyLoss(i, test_entropy, 'test')

	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75, is_training: True})
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
print("test accuracy %g"%accuracy.eval(feed_dict={x: vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, is_training: False}))

saver = tf.train.Saver()
save_path = saver.save(sess, FLAGS.save_path)
