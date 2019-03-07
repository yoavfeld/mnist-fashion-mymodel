from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model
import numpy as np
import plot
import time

def modes(mode, model, weightsFile, x_test, y_test):
    model.load_weights(weightsFile)
    if mode == 'test':
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n', 'Test accuracy:', score[1])
    elif mode == 'pred':
        a = np.full((1, 28, 28, 1), 0)
        a[0] = x_test[0]
        start = time.clock()
        model.predict(a)
        end = time.clock()
        print("Time per image prediction: {} ".format((end - start) / len(x_test)))
    elif mode == 'arc':
        plot_model(model, to_file='model.png', show_shapes=True)
    elif mode == 'vis':
        plot.visualize_accuracy(model, x_test, y_test)
    elif mode == 'cm':
        ypred_onehot = model.predict(x_test)
        ypred = np.argmax(ypred_onehot, axis=1)
        ytrue = np.argmax(y_test, axis=1)

        # compute and plot the confusion matrix
        confusion_mtx = confusion_matrix(ytrue, ypred)
        plot.plot_confusion_matrix(confusion_mtx)

