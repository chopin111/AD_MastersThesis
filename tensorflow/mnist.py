import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)

print("Type of mnist %s" % type(mnist))
print(mnist.train.num_examples)
print(mnist.test.num_examples)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.train.images
testlabel = mnist.train.labels

print("typeof mnist.trian.images %s", type(trainimg))
print("shape of mnist.trian.images %s", trainimg.shape)
print("typeof mnist.trian.labels %s", type(trainlabel))
print("shape of mnist.trian.labels %s", trainlabel.shape)

print("typeof mnist.test.images %s", type(testimg))
print("shape of mnist.test.labels %s", testimg.shape)
print("typeof mnist.test.label %s", type(testlabel))
print("shape of mnist.test.labels %s", testlabel.shape)

def plotNumber(trainimg, idx):
    nsample = 1

    curr_img = np.reshape(trainimg[idx, :], (28, 28))
    curr_label = np.argmax(trainlabel[idx, :])
    plt.matshow(curr_img, cmap = plt.get_cmap('gray'))
    plt.title("" + str(idx) + " Training data. Label is " + str(curr_label))
    print("" + str(idx) + " Training data. Label is " + str(curr_label))
    plt.show()


plotNumber(trainimg, 1)
