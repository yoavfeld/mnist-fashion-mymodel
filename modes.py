from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model
import numpy as np
import plot
import time

def modes(mode, model, x_test, y_test):
    model.load_weights("model.weights.best.hdf5")
    if mode == 'test':
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n', 'Test accuracy:', score[1])
    elif mode == 'arc':
        plot_model(model, to_file='model.png')
    elif mode == 'vis':
        plot.visualize_accuracy(model, x_test, y_test)
    elif mode == 'pred':
        start = time.clock()
        model.predict(x_test)
        end = time.clock()
        print("Time per image: {} ".format((end - start) / len(x_test)))
    elif mode == 'cm':
        ypred_onehot = model.predict(x_test)
        ypred = np.argmax(ypred_onehot, axis=1)
        ytrue = np.argmax(y_test, axis=1)

        # compute and plot the confusion matrix
        confusion_mtx = confusion_matrix(ytrue, ypred)
        plot.plot_confusion_matrix(confusion_mtx)

