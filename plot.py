import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Plot metrics history
def plot_hist(history_file):
    prefix = str(history_file).rsplit('.', maxsplit=1)[0]
    df = pd.read_csv(str(history_file))
    epoch = df['epoch']
    for metric in ['Loss', 'Acc']:
        train = df[metric.lower()]
        val = df['val_' + metric.lower()]
        plt.figure()
        plt.plot(epoch, train, label='train')
        plt.plot(epoch, val, label='val')
        plt.legend(loc='best')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.savefig('.'.join([prefix, metric.lower(), 'png']))
        plt.close()


def visualize(model, x_test, y_test):
    ## Visualize prediction
    y_hat = model.predict(x_test)

    # Define the text labels
    fashion_mnist_labels = ["T-shirt/top",  # index 0
                            "Trouser",      # index 1
                            "Pullover",     # index 2
                            "Dress",        # index 3
                            "Coat",         # index 4
                            "Sandal",       # index 5
                            "Shirt",        # index 6
                            "Sneaker",      # index 7
                            "Bag",          # index 8
                            "Ankle boot"]   # index 9


    # Plot a random sample of 10 test images, their predicted labels and ground truth
    figure = plt.figure(figsize=(20, 8))
    for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
        print('yoav')
        ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        # Display each image
        ax.imshow(np.squeeze(x_test[index]))
        predict_index = np.argmax(y_hat[index])
        true_index = np.argmax(y_test[index])
        # Set the title for each image
        ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
                                      fashion_mnist_labels[true_index]),





                                      color=("green" if predict_index == true_index else "red"))

        #figure.show()
