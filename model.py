import sys
import os
from sklearn.utils import shuffle
from numpy.random import seed

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from tensorflow import set_random_seed
from tensorflow.keras.utils import plot_model

import plot
import callbacks as cb

# init random seed and turn off annoying warning
seed(1)
set_random_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## training params
batch_size=128
epochs=100
lr=0.0001

## images dimensions
w, h = 28, 28
channels = 1

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Data normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train, y_train = shuffle(x_train, y_train)
# Devide to train and validation sets
(x_train, x_valid) = x_train[10000:], x_train[:10000]
(y_train, y_valid) = y_train[10000:], y_train[:10000]

#(x_train, x_valid) = x_train[1000:10000], x_train[:1000]
#(y_train, y_valid) = y_train[1000:10000], y_train[:1000]
#x_test, y_test = x_test[:1000], y_test[:1000]

# Reshape input data from (28, 28) to (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels (binary class vectors)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

model = tf.keras.Sequential()
initializer = tf.contrib.layers.xavier_initializer()

# CNN
# 2 groups of BatchNormalization + Conv + Maxpooling + Droupout layers
model.add(layers.BatchNormalization(input_shape=(h, w, channels)))
model.add(layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.3))

# Converting to 1D feature Vector
model.add(layers.Flatten())

# FC layers
# 2 groups of BatchNormalization, Dense, Dropout.
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu', kernel_initializer=initializer))
model.add(layers.Dropout(0.4))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# BatchNormalization + last dense layer
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=lr)

## Compile the model
model.compile(loss='categorical_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])

# Take a look at the model summary
model.summary()

mode = ""
if len(sys.argv) > 1:
    mode = sys.argv[1]

if mode == 'test':
    # Load the weights with the best validation accuracy
    model.load_weights('model.weights.best.hdf5')
elif mode == 'arc':
    model.load_weights("model.weights.best.hdf5")
    plot_model(model, to_file='model.png')

else:
    ## callbacks
    cb_checkpoint = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=0, save_best_only=True)
    cb_logger = CSVLogger('training.log')
    cb_val_los_monitor = EarlyStopping(monitor='val_loss', patience=10)
    cb_board = TensorBoard(write_grads=True, histogram_freq=5)
    callbacks = [
        cb_checkpoint,
        cb_logger,
        cb_val_los_monitor,
        cb.PlotCB(),
        cb.TestCB((x_test, y_test))
        #cb_board
    ]

    ## Train the model
    model.fit(x_train,
             y_train,
             batch_size=batch_size,
             epochs=epochs,
             validation_data=(x_valid, y_valid),
             callbacks=callbacks)

    plot.plot_hist("training.log")


# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

plot.visualize(model, x_test, y_test)

##TODO (lesson 6 ppt):
# add data normalize:  -mean  / std between pics calculations #37
