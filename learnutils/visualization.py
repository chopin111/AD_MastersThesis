import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

import numpy as np

import random
import time
import csv

plt.style.use(["ggplot"])

axis = 0

class Telemetry:
    def __init__(self, filename="telemetry.csv"):
        self.fieldnames = ['step', 'train_accuracy', 'test_accuracy', 'train_cross_entropy_loss', 'test_cross_entropy_loss', 'step_execution_time']
        self.filename = filename

    def loadAndVisualize(self):
        visualization = DashboardVis(0, 0)
        self.csvfile = open(self.filename, 'r')

        self.csvreader = csv.DictReader(self.csvfile, fieldnames=self.fieldnames)
        next(self.csvreader)
        for row in self.csvreader:
            iteration = int('0' + row['step'])
            test_accuracy = float('0' + row['test_accuracy'])
            train_accuracy = float('0' + row['train_accuracy'])
            train_cross_entropy_loss = float('0' + row['train_cross_entropy_loss'])
            test_cross_entropy_loss = float('0' + row['test_cross_entropy_loss'])

            if train_accuracy:
                visualization.addAccuracy(iteration, train_accuracy, type='train')
            if test_accuracy:
                visualization.addAccuracy(iteration, test_accuracy, type='test')
            if train_cross_entropy_loss:
                visualization.addCrossEntropyLoss(iteration, train_cross_entropy_loss, type='train')
            if test_cross_entropy_loss:
                visualization.addCrossEntropyLoss(iteration, test_cross_entropy_loss, type='test')

        visualization.update()
        visualization.draw()
        visualization.save()


    def create(self):
        self.csvfile = open(self.filename, 'w')

        self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
        self.csvwriter.writeheader()

    def addAccuracy(self, iteration, value, type='train'):
        accuracy_type = 'test_accuracy'
        if type is 'train':
            accuracy_type = 'train_accuracy'
        values = {'step': iteration}
        values[accuracy_type] = value
        self.csvwriter.writerow(values)
        self.csvfile.flush()

    def addCrossEntropyLoss(self, iteration, value, type='train'):
        accuracy_type = 'test_cross_entropy_loss'
        if type is 'train':
            accuracy_type = 'train_cross_entropy_loss'
        values = {'step': iteration}
        values[accuracy_type] = value
        self.csvwriter.writerow(values)
        self.csvfile.flush()

    def addStepExecutionTime(self, iteration, steptime):
        self.csvwriter.writerow({'step': iteration, 'step_execution_time': steptime})
        self.csvfile.flush()

class DashboardVis:
    def __init__(self, imageWidth, imageHeight):
        self.fig = plt.figure()
        plt.gcf().canvas.set_window_title("fMri")
        self.fig.set_facecolor('#FFFFFF')

        self.x1 = []
        self.y1 = []
        self.x2 = []
        self.y2 = []
        self.x3 = []
        self.y3 = []
        self.x4 = []
        self.y4 = []

        self.imageGrayscale = np.full((imageWidth, imageHeight, 40), 1, dtype='float32')

        self.iterations = 0
        self.maxAccuracy = 1
        self.maxSlice = 39
        self.minSlice = 0
        self.sliceInd = 20
        self.maxCrossEntropyLoss = 1

        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        #ax4 = fig.add_subplot(234)
        #ax5 = fig.add_subplot(235)
        #ax6 = fig.add_subplot(236)

        self.line1, = self.ax1.plot(self.x1, self.y1, label="training accuracy")
        self.line2, = self.ax1.plot(self.x2, self.y2, label="test accuracy")

        self.line3, = self.ax2.plot(self.x3, self.y3, label="training loss")
        self.line4, = self.ax2.plot(self.x4, self.y4, label="test loss")

        self.ax1.set_title("Accuracy", y=1.02)
        self.ax1.legend(loc='upper left')
        self.ax2.set_title("Cross entropy loss", y=1.02)
        self.ax2.legend(loc='upper left')
        self.ax3.set_title("Train image (1 of 10)", y=1.02)

        self.ax3.grid(False)
        self.ax3.set_axis_off()

        #self.image1 = self.ax3.imshow(self.imageGrayscale, cmap='gray', origin='lower')
        self.image1 = self.ax3.imshow(self.imageGrayscale[:, :, 20].T, cmap='gray', origin="lower")

    def updateSliceInd(self, percentage):
        maxSlice = self.imageGrayscale.shape[axis] - 1

        if self.sliceInd+1 > maxSlice:
            self.sliceInd = 0
        else:
            self.sliceInd = int(maxSlice*percentage)
        self.update()


    def addAccuracy(self, iteration, value, type='train'):
        if type is 'train':
            self.x1.append(iteration)
            self.y1.append(value)
        else:
            self.x2.append(iteration)
            self.y2.append(value)
        self.iterations = max(self.iterations, iteration)
        self.maxAccuracy = max(self.maxAccuracy, value)

    def addCrossEntropyLoss(self, iteration, value, type='train'):
        if type is 'train':
            self.x3.append(iteration)
            self.y3.append(value)
        else:
            self.x4.append(iteration)
            self.y4.append(value)
        self.iterations = max(self.iterations, iteration)

        self.maxCrossEntropyLoss = max(self.maxCrossEntropyLoss, value)

    def updateImage(self, image):
        self.imageGrayscale = image

    def update(self):
        self.line1.set_data(self.x1, self.y1)
        self.line2.set_data(self.x2, self.y2)

        self.line3.set_data(self.x1, self.y1)
        self.line4.set_data(self.x2, self.y2)

        self.ax1.set_xlim(0, self.iterations+1)
        self.ax2.set_xlim(0, self.iterations+1)

        self.ax1.set_ylim(0, self.maxAccuracy)
        self.ax2.set_ylim(0, 1)

        #self.image1.set_data(self.imageGrayscale.T)
        if self.imageGrayscale:
            imSlice = np.take(self.imageGrayscale, self.sliceInd, axis=axis)
            self.image1 = self.ax3.imshow(imSlice.T, cmap='gray', origin="lower")

    def draw(self):
        plt.pause(0.001)
        plt.show(block=False)

    def save(self):
        self.fig.set_size_inches(150, 18.5)
        extent = self.ax1.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig('ax1_figure.png', bbox_inches=extent.expanded(1.1, 1.2))

        extent = self.ax2.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig('ax2_figure.png', bbox_inches=extent.expanded(1.1, 1.2))

if __name__ == '__main__':
    telemtry = Telemetry()
    telemtry.loadAndVisualize()
