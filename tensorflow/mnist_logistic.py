import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.train.images
testlabel = mnist.train.labels

learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

actv = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))

optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(pred, "float"))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        num_batch = int(mnist.train.num_examples/batch_size)

        for i in range(num_batch):
            if 0:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            else:
                randidx = np.random.randint(trainimg.shape[0], size=batch_size)

                batch_xs = trainimg[randidx, :]
                batch_ys = trainlabel[randidx, :]

            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch

        if epoch % display_step == 0:
            train_acc = accr.eval({x: batch_xs, y: batch_ys})
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f" % (epoch, training_epochs, avg_cost, train_acc))

    print ("Accuracy: ", accr.eval({x: batch_xs, y: batch_ys}))
