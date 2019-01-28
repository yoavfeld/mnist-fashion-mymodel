import plot
from tensorflow import keras

class PlotCB(keras.callbacks.Callback):
    def on_epoch_end(self, logs={}, c={}):
        plot.plot_hist("training.log")

    #def on_batch_end(self, batch, logs={}):


class TestCB(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
