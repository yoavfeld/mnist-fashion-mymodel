
from tensorflow.keras.utils import plot_model
import plot
import time

def modes(mode, model, x_test, y_test):
    if mode == 'test':
        # Load the weights with the best validation accuracy
        model.load_weights('model.weights.best.hdf5')
        print('\n', 'Test accuracy:', score[1])
    elif mode == 'arc':
        model.load_weights("model.weights.best.hdf5")
        plot_model(model, to_file='model.png')
    elif mode == 'vis':
        model.load_weights("model.weights.best.hdf5")
        plot.visualize(model, x_test, y_test)
    elif mode == 'pred':
        start = time.clock()
        model.predict(x_test)
        end = time.clock()
        print("Time per image: {} ".format((end - start) / len(x_test)))