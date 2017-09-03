import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import numpy as np

import random
import time

plt.style.use(["ggplot"])

class DashboardVis:
    def __init__(self):
        fig = plt.figure()
        plt.gcf().canvas.set_window_title("fMri")
        fig.set_facecolor('#FFFFFF')

        self.x1 = []
        self.y1 = []
        self.x2 = []
        self.y2 = []
        self.x3 = []
        self.y3 = []
        self.x4 = []
        self.y4 = []
        self.iterations = 0
        self.maxAccuracy = 1
        self.maxCrossEntropyLoss = 10

        self.ax1 = fig.add_subplot(131)
        self.ax2 = fig.add_subplot(132)
        self.ax3 = fig.add_subplot(133)
        #ax4 = fig.add_subplot(234)
        #ax5 = fig.add_subplot(235)
        #ax6 = fig.add_subplot(236)

        self.line1, = self.ax1.plot(self.x1, self.y1, label="training accuracy")
        self.line2, = self.ax1.plot(self.x2, self.y2, label="test accuracy")

        self.line3, = self.ax2.plot(self.x3, self.y3, label="training loss")
        self.line4, = self.ax2.plot(self.x4, self.y4, label="test loss")

        self.ax1.set_title("Accuracy", y=1.02)
        self.ax2.set_title("Cross entropy loss", y=1.02)
        self.ax3.set_title("Train image (1 of 10)", y=1.02)

    def addAccuracy(self, iteration, value, type='train'):
        if type is 'train':
            self.x1.append(iteration)
            self.y1.append(value)
        else:
            self.x2.append(iteration)
            self.y2.append(value)
        self.iterations = max(self.iterations, iteration)


    def addCrossEntropyLoss(self, iteration, value, type='train'):
        if type is 'train':
            self.x3.append(iteration)
            self.y3.append(value)
        else:
            self.x4.append(iteration)
            self.y4.append(value)
        self.iterations = max(self.iterations, iteration)

    def update(self):
        self.line1.set_data(self.x1, self.y1)
        self.line2.set_data(self.x2, self.y2)

        self.line3.set_data(self.x1, self.y1)
        self.line4.set_data(self.x2, self.y2)

        self.ax1.set_xlim(0, self.iterations+1)
        self.ax2.set_xlim(0, self.iterations+1)

        self.ax1.set_ylim(0, self.maxAccuracy)
        self.ax2.set_ylim(0, self.maxCrossEntropyLoss)

    def draw(self):
        plt.pause(0.001)
        plt.show(block=False)

if __name__ == '__main__':
    # just a test
    vis = DashboardVis()
    vis.update()
    #vis.draw()
    for i in range(0, 100):
        print(i)
        vis.addAccuracy(i, random.random(), type='train')
        vis.addCrossEntropyLoss(i, random.random()*10, type='train')
        vis.update()
        if i % 10:
            vis.addAccuracy(i, random.random(), type='test')
            vis.addCrossEntropyLoss(i, random.random()*10, type='test')
            vis.draw()
            time.sleep(2)
