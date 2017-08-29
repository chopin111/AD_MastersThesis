width = 128
height = 128
depth = 40
nLabel = 31

# Start TensorFlow InteractiveSession
import gc
import tensorflow as tf
import numpy as np

######################################
######################################
######################################
######      DATA LOADING        ######
######################################
######################################
######################################


nLabel = 31

FLAGS = tf.app.flags.FLAGS
FLAGS.width = 128
FLAGS.height = 128
FLAGS.depth = 40 # 3

# Function to tell TensorFlow how to read a single image from input file
def getImage(filename):
	# convert filenames to a queue for an input pipeline.
	filenameQ = tf.train.string_input_producer([filename],num_epochs=None)

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

label, image = getImage("data-tensorflow/train-00000-of-00001")
vlabel, vimage = getImage("data-tensorflow/validation-00000-of-00001")

imageBatch, labelBatch = tf.train.shuffle_batch(
	[image, label], batch_size=100,
	capacity=2000,
	min_after_dequeue=1000)

vimageBatch, vlabelBatch = tf.train.shuffle_batch(
	[vimage, vlabel], batch_size=100,
	capacity=2000,
	min_after_dequeue=1000)

######################################
######################################
######################################
######    END DATA LOADING      ######
######################################
######################################
######################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#with tf.device('/gpu:0'):
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
#withend

sess = tf.InteractiveSession(config=config)
sess.run(c)
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
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
	return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')


## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool
print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32


## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64]) # [64]

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64
gc.collect()

## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([8*8*3*64, 512])  # [7*7*64, 512]
b_fc1 = bias_variable([512]) # [512]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*3*64])  # -> output image: [-1, 7*7*64] = 3136
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

# Include keep_prob in feed_dict to control dropout rate.
for i in range(35):
	print("step %d"%(i))
	batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
	# Logging every 100th iteration in the training process.
	if i%5 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
	gc.collect()

# Evaulate our accuracy on the test data
vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
print("test accuracy %g"%accuracy.eval(feed_dict={x: vbatch_xs, y_: vbatch_ys, keep_prob: 1.0}))
