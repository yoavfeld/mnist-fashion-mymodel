import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

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

def visualize_accuracy(model, x_test, y_test):
    y_hat = model.predict(x_test)

    # Calc false predictions precentage per label
    false_predictions = [0] * 10
    total_predictions = [0] * 10


    for index in range(0,10000):
        predict_index = np.argmax(y_hat[index])
        true_index = np.argmax(y_test[index])
        total_predictions[true_index] += 1
        if predict_index != true_index:
            false_predictions[true_index] += 1

    # prints 10 labels with its false predictions precentage
    figure = plt.figure(figsize=(20, 8))
    for i in range(0, 10):
        ax = figure.add_subplot(2, 5, i + 1, xticks=[], yticks=[])

        # Display the current image
        for j in range(0, 1000):
            if np.argmax(y_test[j]) == i:
                ax.imshow(np.squeeze(x_test[j]))
                break

        # Set the title for each image
        suc_rate = 100-((false_predictions[i]/total_predictions[i])*100)
        ax.set_title("{}\n{} ({}%)\n{} ({})".format(fashion_mnist_labels[i],
                                            "Success: ",
                                            float("{0:.2f}".format(suc_rate)),
                                            "Total: ",
                                            total_predictions[i]),
                                            color=("green"))

    plt.show()

def visualize_sample(model, x_test, y_test):
    ## Visualize prediction
    y_hat = model.predict(x_test)

    # Plot a random sample of 10 test images, their predicted labels and ground truth
    figure = plt.figure(figsize=(20, 8))
    for i,index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):

        ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        # Display each image
        ax.imshow(np.squeeze(x_test[index]))
        predict_index = np.argmax(y_hat[index])
        true_index = np.argmax(y_test[index])
        # Set the title for each image
        ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
                                      fashion_mnist_labels[true_index]),
                     color=("green" if predict_index == true_index else "red"))

    plt.show()

def plot_confusion_matrix(cm, classes=fashion_mnist_labels, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()