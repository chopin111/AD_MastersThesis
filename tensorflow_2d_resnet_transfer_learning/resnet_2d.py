import gc
import tensorflow as tf
import numpy as np

import os
import io

import random
import time
from datetime import date

from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import inception_preprocessing

from tensorflow.python.platform import tf_logging as logging

from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step


slim = tf.contrib.slim

# Path hack.
import sys
sys.path.insert(0, os.path.abspath('..'))

######################################
######################################
######################################
######	  DATA LOADING		######
######################################
######################################
######################################


#State where your log file is at. If it doesn't exist, create it.
log_dir = './log'


nLabel = 5

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 3
FLAGS.height = 224
FLAGS.depth = 224 # 3

FLAGS.iterations = 20000 # 3

#FLAGS.batch_size = 50

FLAGS.train_update_count = 10
FLAGS.test_update_count = 100
FLAGS.checkpoint = 1000

#TODO: change those
#State the number of epochs to train
num_epochs = 100

#State your batch size
batch_size = 50

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.5
num_epochs_before_decay = 1

FLAGS.source_model_path = './inception_resnet_v2_2016_08_30.ckpt'

# Function to tell TensorFlow how to read a single image from input file
def getImage(filenames, is_training):
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
	label = tf.cast(label, tf.int64)

	image_buffer = features['image/encoded']

	image = tf.decode_raw(image_buffer, tf.float32)
	image = tf.reshape(image, [FLAGS.width,FLAGS.depth,FLAGS.height])
	image = tf.transpose(image, perm=[1, 2, 0]) # move channels to back
	#image = tf.reshape(image, tf.stack([FLAGS.width*FLAGS.height*FLAGS.depth]))

	print(image.shape)
	#image = inception_preprocessing.preprocess_image(image, FLAGS.height, FLAGS.width, is_training)
	#print(image.shape)

	#label=tf.stack(tf.one_hot(label-1, nLabel))
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

with tf.Graph().as_default() as graph:
	tf.logging.set_verbosity(tf.logging.INFO)

	label, image = getImage(train_files, True)
	vlabel, vimage = getImage(validation_files, False)

	imageBatch, labelBatch = tf.train.shuffle_batch(
		[image, label], batch_size=batch_size,
		capacity=1000,
		min_after_dequeue=100)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	sess = tf.InteractiveSession(config=config)

	num_samples = 2700*4 #TODO

	with slim.arg_scope(inception_resnet_v2_arg_scope()):
		logits, end_points = inception_resnet_v2(imageBatch, num_classes = nLabel, is_training = True)

	exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
	variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

	one_hot_labels = slim.one_hot_encoding(labelBatch, nLabel)

	#loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
	#total_loss = tf.losses.get_total_loss()
	loss = slim.losses.softmax_cross_entropy(logits, one_hot_labels) # TODO: zamienilem dwie linijki wyzej na to
	total_loss = slim.losses.get_total_loss()

	num_batches_per_epoch = int(num_samples / batch_size)
	num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
	decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

	global_step = get_or_create_global_step()
	lr = tf.train.exponential_decay(
				learning_rate = initial_learning_rate,
				global_step = global_step,
				decay_steps = decay_steps,
				decay_rate = learning_rate_decay_factor,
				staircase = True)

	optimizer = tf.train.AdamOptimizer(learning_rate = lr)

	train_op = slim.learning.create_train_op(total_loss, optimizer)

	predictions = tf.argmax(end_points['Predictions'], 1)
	probabilities = end_points['Predictions']
	accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labelBatch)
	metrics_op = tf.group(accuracy_update, probabilities)

	tf.summary.scalar('losses/Total_Loss', total_loss)
	tf.summary.scalar('accuracy', accuracy)
	tf.summary.scalar('learning_rate', lr)
	my_summary_op = tf.summary.merge_all()

 	#Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
	def train_step(sess, train_op, global_step):
		'''
		Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
		'''
		#Check the time for each sess run
		start_time = time.time()
		total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
		time_elapsed = time.time() - start_time

		#Run the logging to print some results
		logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

		return total_loss, global_step_count

	#Now we create a saver function that actually restores the variables from a checkpoint file in a sess
	saver = tf.train.Saver(variables_to_restore)
	def restore_fn(sess):
		return saver.restore(sess, FLAGS.source_model_path)

	#Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory

	sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)

	#Run the managed session
	with sv.managed_session() as sess:

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)

		for step in range(num_steps_per_epoch * num_epochs):
			#At the start of every epoch, show the vital information:
			if step % num_batches_per_epoch == 0:
				logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
				learning_rate_value, accuracy_value = sess.run([lr, accuracy])
				logging.info('Current Learning Rate: %s', learning_rate_value)
				logging.info('Current Streaming Accuracy: %s', accuracy_value)

				# optionally, print your logits and predictions for a sanity check that things are going fine.
				logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labelBatch])
				print('logits: %s\n', logits_value)
				print('Probabilities: %s\n', probabilities_value)
				print('predictions: %s\n', predictions_value)
				print('Labels: %s' % labels_value)

			#Log the summaries every 10 step.
			if step % 10 == 0:
				loss, _ = train_step(sess, train_op, sv.global_step)
				summaries = sess.run(my_summary_op)
				sv.summary_computed(sess, summaries)

			#If not, simply run the training step
			else:
				loss, _ = train_step(sess, train_op, sv.global_step)

		#We log the final training loss and accuracy
		logging.info('Final Loss: %s', loss)
		logging.info('Final Accuracy: %s', sess.run(accuracy))

		#Once all the training has been done, save the log files and checkpoint model
		logging.info('Finished training! Saving model to disk now.')
		# saver.save(sess, "./flowers_model.ckpt")
		sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
