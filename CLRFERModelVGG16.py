# USAGE
# python CLRFERModelVGG16.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Packages import
import os
import cv2
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.minigooglenet import MiniGoogLeNet
from pyimagesearch.clr_callback import CyclicLR
from pyimagesearch import config

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop,adam
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import callbacks

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 200

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
                help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())


# Read images

# Define data path
data_path = './mergedDataset/'
data_dir_list = os.listdir(data_path)

# img_rows = 256
# img_cols = 256
# num_channel = 1
#
# num_epoch = 10

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loading images from dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128, 128))
        # img_data_list.append(input_img_resize)
        img_data_list.append(input_img_resize)

    print('Loaded images from dataset-' + '{}\n'.format(dataset))

img_data = np.array(img_data_list)
# print(img_data, ":first.")
img_data = img_data.astype('float32')
# print(img_data, ":second.")
img_data = img_data / 255
# print(img_data, ":third.")
print(img_data.shape)

# Define the number of classes

num_classes = 7

num_of_samples = img_data.shape[0]
print(num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:75] = 0  # 75
labels[75:162] = 1  # 87
labels[162:219] = 2  # 57
labels[219:319] = 3  # 100
labels[319:378] = 4  # 59
labels[378:491] = 5  # 113
labels[491:] = 6  # 148


print(len(labels))

names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def get_label(id):
    return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][id]


# Class label to one-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
# print(Y)
# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# print(x, "\n", y)
# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=1)

# print(len(X_train), " and\n", len(X_test))
# print(len(Y_train), " and\n", len(Y_test))
# CNN layer set

# Data augmentation
aug = ImageDataGenerator()
#  use fit.generate when fitting data
# datagen = ImageDataGenerator(horizontal_flip=True)
# datagen.fit(X_train)


input_shape = img_data[0].shape
print(input_shape)
# print(img_data)


# Defining the model
model = Sequential()

# VGG
# Feature Extraction
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile Model
opt = adam(lr=config.MIN_LR)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# View model configuration
model.get_weights()

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])

# Model Save
# model.save_weights('model_weights_ckjaffe_256px_Kim_CLR_256E_R?.h5')
# model.save('model_keras_merged_128px_VGG_v1.h5')

# Model Load
# from keras.models import load_model
# model = load_model('model_keras_ckjaffe_256px_Kim_Drop_L2_150E.h5')
# model.load_weights('./tmp/Best-weights-my_model-104-0.2674-0.9783.hdf5')


# check to see if we are attempting to find an optimal learning rate
# before training for the full number of epochs
if args["lr_find"] > 0:
    # initialize the learning rate finder and then train with learning
    # rates ranging from 1e-10 to 1e+1
    print("[INFO] finding learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(
        aug.flow(X_train, Y_train, batch_size=config.BATCH_SIZE),
        1e-10, 1e+1,
        stepsPerEpoch=np.ceil((len(X_train) / float(config.BATCH_SIZE))),
        batchSize=config.BATCH_SIZE)

    # plot the loss for the various learning rates and save the
    # resulting plot to disk
    lrf.plot_loss()
    plt.savefig(config.LRFIND_PLOT_PATH)

    # gracefully exit the script so we can adjust our learning rates
    # in the config and then train the network for our full set of
    # epochs
    print("[INFO] learning rate finder complete")
    print("[INFO] examine plot and adjust learning rates before training")
    sys.exit(0)

# Export training data to a csv file
filename = 'model_train_vgg_merged.csv'
csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

# Export best weights
fp = "./tmp/CLR_Triangular/Best-weights-my_model-{epoch:03d}-{val_acc:.4f}.hdf5"
checkpoint_weight = callbacks.ModelCheckpoint(filepath=fp, verbose=1, save_best_only=True)

# otherwise, we have already defined a learning rate space to train
# over, so compute the step size and initialize the cyclic learning
# rate method
stepSize = config.STEP_SIZE * (X_train.shape[0] // config.BATCH_SIZE)

clr = CyclicLR(
    mode=config.CLR_METHOD,
    base_lr=config.MIN_LR,
    max_lr=config.MAX_LR,
    step_size=stepSize
    )

print("Step size: ", stepSize)
callbacks_list = [csv_log, checkpoint_weight, clr]

# train the network
print("[INFO] training network...")
# Model training
start_time = time.time()

# hist = model.fit(X_train, Y_train,
#                  batch_size=64,
#                  epochs=150,
#                  verbose=1,
#                  validation_split=0.15,
#                  callbacks=callbacks_list)

H = model.fit(
    X_train, Y_train,
    batch_size=config.BATCH_SIZE,
    validation_split=0.15,
    epochs=config.NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=config.BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1),
      predictions.argmax(axis=1), target_names=config.CLASSES))
print("-- %s seconds --" % (time.time() - start_time))


# use cnn test images in Oja

# construct a plot that plots and saves the training history
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)

# Confusion Matrix
results = model.predict_classes(X_test)
# cm = confusion_matrix(np.where(Y_test == 1)[1], results)
y_trueskl = np.where(Y_test == 1)[1]
y_predskl = results

y_truelbl = pandas.Series(np.where(Y_test == 1)[1], name="True")
y_predlbl = pandas.Series(results, name="Predicted")

cm_pand = pandas.crosstab(y_truelbl, y_predlbl)
print(cm_pand)

cm_pand.to_csv('conf_mtx_vgg_merged.csv')


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.gray_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            print('Normalized confusion matrix')
        else:
            print('Confusion matrix, without normalization')

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = names  # list(unique_labels(y_true, y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.grid(False)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Save non-normalized confusion matrix
plot_confusion_matrix(y_trueskl, y_predskl)
plt.savefig(os.path.sep.join(["output", "conf_mtx_vgg_merged.png"]))

# Save normalized confusion matrix
plot_confusion_matrix(y_trueskl, y_predskl, normalize=True)
plt.savefig(os.path.sep.join(["output", "conf_mtx_vgg_merged_norm.png"]))
