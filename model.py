import sys
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from tensorflow import set_random_seed
from numpy.random import seed

import modes
import callbacks as cb

# init random seed (enable for development)
#seed(1)
#set_random_seed(1)

#turn off annoying warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## params
batch_size=128
epochs=150
initializer = tf.contrib.layers.xavier_initializer()
lr=0.0001
activation = 'relu'
#act_layer = layers.LeakyReLU()
weightsFile = 'model.weights.best.hdf5'

## images dimensions
w, h = 28, 28
channels = 1

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Data normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Devide to train and validation sets
(x_train, x_valid) = x_train[10000:], x_train[:10000]
(y_train, y_valid) = y_train[10000:], y_train[:10000]

# Reshape input data from (28, 28) to (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels (binary class vectors)
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

model = keras.Sequential()

# CNN - 2 X BatchNormalization + Conv + Maxpooling + Droupout layers
model.add(layers.BatchNormalization(input_shape=(h, w, channels)))
model.add(layers.Conv2D(filters=32,kernel_size=3,activation=activation))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64,kernel_size=3,padding='same',activation=activation))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.3))

# Converting to 1D feature Vector
model.add(layers.Flatten())

# FC - 2 X BatchNormalization, Dense, Dropout.
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation=activation, kernel_initializer=initializer))
model.add(layers.Dropout(0.4))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation=activation))
model.add(layers.Dropout(0.5))

# BatchNormalization + last dense layer
model.add(layers.BatchNormalization())
model.add(layers.Dense(10))

opt = keras.optimizers.Adam(lr=lr)

## Compile the model
model.compile(loss='categorical_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])

# check mode argument
if len(sys.argv) > 1:
    mode = sys.argv[1]
    modes.modes(mode, model, weightsFile, x_test, y_test)
    sys.exit()

# Take a look at the model summary
model.summary()

cb_checkpoint = ModelCheckpoint(filepath=weightsFile, verbose=0, save_best_only=True)
cb_logger = CSVLogger('training.log')
cb_val_los_monitor = EarlyStopping(monitor='val_loss', patience=10)
callbacks = [
    cb_checkpoint,
    cb_logger,
    cb_val_los_monitor,
    cb.PlotCB(),
    cb.TestCB((x_test, y_test)),
]

## Train the model
model.fit(x_train,
         y_train,
         batch_size=batch_size,
         epochs=epochs,
         validation_data=(x_valid, y_valid),
         callbacks=callbacks)


# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
