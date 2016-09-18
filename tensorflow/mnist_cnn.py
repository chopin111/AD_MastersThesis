#UWAGA! program testowy, nie do końcowej wersji magisterki

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

trainimg    = mnist.train.images
trainlabel  = mnist.train.labels
testimg     = mnist.train.images
testlabel   = mnist.train.labels

n_input = 784
n_output = 10

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([14*14*64, n_output], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bd1':  tf.Variable(tf.random_normal([n_output], stddev=0.1)),
}

def conv_simple(_input, _w, _b):
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1]) #reshape input
    _conv1  = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME') #Convolution
    _conv2  = tf.nn.bias_add(_conv1, _b['bc1']) #add bias
    _conv3  = tf.nn.relu(_conv2) #pass relu
    _pool   = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max-pooling
    _dense  = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]]) #vectorize
    _out    = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1']) #fully-connected layer

    out = {
        'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv2': _conv3,
        'pool': _pool, 'dense': _dense, 'out': _out
    }
    return out


#input_data
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
#parameters
learning_rate   = 0.001
training_epochs = 5
batch_size      = 100
display_step    = 1

_pred   = conv_simple(x, weights, biases)['out']
cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
optm    = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
_corr   = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1)) # Count corrects
accr   = tf.reduce_mean(tf.cast(_corr, tf.float32)) #Accuracy
init    = tf.initialize_all_variables()

# Saver
save_step = 1
savedir = "nets/"
saver = tf.train.Saver(max_to_keep=training_epochs)


do_train = 1

sess = tf.Session()
sess.run(init)

if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            #Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})

            #Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch

        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
            print (" Training accuracy: %.3f" % (train_acc))
            train_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
            print (" Test accuracy: %.3f" % (test_acc))

        if epoch % save_step == 0:
            saver.save(sess, "nets/cnn_simple.ckpt-" + str(epoch))

# Now let's see what all variables looks like
conv_out = conv_simple(x, weight, biases)
