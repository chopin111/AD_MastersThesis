#### Week3-1c What is CNN (cnn_mnist_simple).pptx
#### NOTE!!!! PRZEROBIC, BO TO PRZYKLAD!!!
####

####
#### IMPORTS
####
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import input_numpy

print ("Packs loaded")

####
#### CNN structure
####

# Network
#with tf.device('/gpu:0'):
n_input = 784
n_output = 31
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}

def conv_basic(_input, _w, _b, _keepratio):
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1]) # Reshape input
    # conv1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME') # Convolutioin
    _conv1 = tf.nn.batch_normalization(_conv1, 0.001, 1.0, 0, 1, 0.0001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1'])) # Add-bias
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # Max-pooling
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # conv1
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME') # Convolutioin
    _conv2 = tf.nn.batch_normalization(_conv2, 0.001, 1.0, 0, 1, 0.0001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2'])) # Add-bias
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # Max-pooling
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    #vectorize
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    #fc1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    #fc2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    out = {
        'input_r': _input_r,
        'conv1': _conv1,
        'pool1': _pool1,
        'pool_dr1': _pool_dr1,
        'conv2': _conv2,
        'pool2': _pool2,
        'pool_dr2': _pool_dr2,
        'dense1': _dense1,
        'fc1': _fc1,
        'fc_dr1': _fc_dr1,
        'out': _out
    } # Return everything
    return out
#end tf.device('/gpu:0'):

print ("CNN ready")



####
#### Define functions
####

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32, name="CNN_DROPOUT_keepratio")
# Parameters
learning_rate = 0.001
training_epochs = 150
batch_size = 100
display_step = 1
# Functions!
#with tf.device('/gpu:0'):
_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()
#end tf.device('/gpu:0'):


# Saver
save_step = 1
savedir = "nets/"
saver = tf.train.Saver(max_to_keep=training_epochs)
print ("Network Ready to Go!")

####
#### Optimization
####

do_train = 1

# Do some optimizations
sess = tf.Session()
sess.run(init)

train_data = input_numpy.read_inputs(training_epochs, batch_size, './data-train')
test_data = input_numpy.read_inputs(training_epochs, batch_size, './data-test')

if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_data.num_examples/batch_size)
        print total_batch
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train_data.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            print ("%s %s" % (batch_xs, batch_ys))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
            print (" Training accuracy: %.3f" % (train_acc))
            test_acc = sess.run(accr, feed_dict={x: test_data.total_images, y: test_data.total_labels, keepratio:1.})
            print (" Test accuracy: %.3f" % (test_acc))

        # Save Network
        if epoch % save_step == 0:
            saver.save(sess, "nets/cnn_simple.ckpt-" + str(epoch))

    print ("Optimization Finished.")
