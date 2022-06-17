# USAGE
# python CLRFERModelVGG.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib

# Packages import
import os
import cv2
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.minigooglenet import MiniGoogLeNet
from pyimagesearch.clr_callback import CyclicLR
from pyimagesearch import config

from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, adam
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import callbacks

matplotlib.use("Agg")

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

# Define data path for training
data_path_train = './ferBigTrainDataset/'
data_dir_list_train = os.listdir(data_path_train)

img_data_list_train = []

for dataset in sorted(data_dir_list_train):
    img_list_train = os.listdir(data_path_train + '/' + dataset)
    print('Loading images from training dataset-' + '{}\n'.format(dataset))
    for img in img_list_train:
        input_img_train = cv2.imread(data_path_train + '/' + dataset + '/' + img)
        # input_img_train=cv2.cvtColor(input_img_train, cv2.COLOR_BGR2GRAY)
        # input_img_train_resize = cv2.resize(input_img_train, (48, 48))
        # img_data_list.append(input_img_train_resize)
        img_data_list_train.append(input_img_train)

    print('Loaded images from training dataset-' + '{}\n'.format(dataset))

img_data_train = np.array(img_data_list_train)
img_data_train = img_data_train.astype('float32')
img_data_train = img_data_train / 255
print(img_data_train.shape)

# Define data path for validation
data_path_val = './ferBigValDataset/'
data_dir_list_val = os.listdir(data_path_val)

img_data_list_val = []

for dataset in sorted(data_dir_list_val):
    img_list_val = os.listdir(data_path_val + '/' + dataset)
    print('Loading images from validation dataset-' + '{}\n'.format(dataset))
    for img in img_list_val:
        input_img_val = cv2.imread(data_path_val + '/' + dataset + '/' + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # input_img_resize = cv2.resize(input_img, (48, 48))
        # img_data_list.append(input_img_resize)
        img_data_list_val.append(input_img_val)

    print('Loaded images from validation dataset-' + '{}\n'.format(dataset))

img_data_val = np.array(img_data_list_val)
img_data_val = img_data_val.astype('float32')
img_data_val = img_data_val / 255
print(img_data_val.shape)


# Define data path for testing
data_path_test = './ferBigTestDataset/'
data_dir_list_test = os.listdir(data_path_test)

img_data_list_test = []

for dataset in sorted(data_dir_list_test):
    img_list_test = os.listdir(data_path_test + '/' + dataset)
    print('Loading images from testing dataset-' + '{}\n'.format(dataset))
    for img in img_list_test:
        input_img_test = cv2.imread(data_path_test + '/' + dataset + '/' + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # input_img_resize = cv2.resize(input_img, (48, 48))
        # img_data_list.append(input_img_resize)
        img_data_list_test.append(input_img_test)

    print('Loaded images from testing dataset-' + '{}\n'.format(dataset))

img_data_test = np.array(img_data_list_test)
img_data_test = img_data_test.astype('float32')
img_data_test = img_data_test / 255
print(img_data_test.shape)

# Define the number of classes

num_classes = 7

# Define number of samples for training
num_of_samples_train = img_data_train.shape[0]
print(num_of_samples_train)
labels_train = np.ones((num_of_samples_train,), dtype='int64')

labels_train[0:3995] = 0  # 3995
labels_train[3995:4431] = 1  # 436
labels_train[4431:8528] = 2  # 4097
labels_train[8528:15743] = 3  # 7215
labels_train[15743:20573] = 4  # 4830
labels_train[20573:23744] = 5  # 3171
labels_train[23744:] = 6  # 4965

print(len(labels_train))

# Class label to one-hot encoding
Y_tr = np_utils.to_categorical(labels_train, num_classes)

# Shuffle the dataset
x_tr, y_tr = shuffle(img_data_train, Y_tr, random_state=2)
# print(x, "\n", y)

# Assigning shuffled values to a set
X_train, Y_train = x_tr, y_tr
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=2)

# print(len(X_train), " and\n", len(X_test))
# print(len(Y_train), " and\n", len(Y_test))


# Define number of samples for validation
num_of_samples_val = img_data_val.shape[0]
print(num_of_samples_val)
labels_val = np.ones((num_of_samples_val,), dtype='int64')

labels_val[0:467] = 0  # 467
labels_val[467:523] = 1  # 56
labels_val[523:1019] = 2  # 496
labels_val[1019:1914] = 3  # 895
labels_val[1914:2567] = 4  # 653
labels_val[2567:2982] = 5  # 415
labels_val[2982:] = 6  # 607

print(len(labels_val))

# Class label to one-hot encoding
Y_va = np_utils.to_categorical(labels_val, num_classes)
# print(Y)
# Shuffle the dataset
x_va, y_va = shuffle(img_data_val, Y_va, random_state=2)
# print(x, "\n", y)
# Assigning shuffled values to a set
X_val, Y_val = x_va, y_va
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=2)

# print(len(X_train), " and\n", len(X_test))
# print(len(Y_train), " and\n", len(Y_test))


# Define number of samples for testing
num_of_samples_test = img_data_test.shape[0]
print(num_of_samples_test)
labels_test = np.ones((num_of_samples_test,), dtype='int64')

labels_test[0:491] = 0  # 491
labels_test[491:546] = 1  # 55
labels_test[546:1074] = 2  # 528
labels_test[1074:1953] = 3  # 879
labels_test[1953:2547] = 4  # 594
labels_test[2547:2963] = 5  # 416
labels_test[2963:] = 6  # 626

print(len(labels_test))

# Class label to one-hot encoding
Y_te = np_utils.to_categorical(labels_test, num_classes)
# print(Y)
# Shuffle the dataset
x_te, y_te = shuffle(img_data_test, Y_te, random_state=2)
# print(x, "\n", y)
# Assigning shuffled values to a set
X_test, Y_test = x_te, y_te
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=2)

# print(len(X_train), " and\n", len(X_test))
# print(len(Y_train), " and\n", len(Y_test))

names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def get_label(id):
    return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][id]


# CNN layer set
# Data augmentation
aug = ImageDataGenerator()
#  use fit.generate when fitting data
# datagen = ImageDataGenerator(horizontal_flip=True)
# datagen.fit(X_train)


input_shape = img_data_train[0].shape
print(input_shape)
# print(img_data)


# Defining the model
model = Sequential()

# VGG
# Feature Extraction
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
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
# model.save('model_keras_fer2013_vgg_CLR_256E_r?.h5')

# Model Load
# from keras.models import load_model
# model = load_model('model_keras_ckjaffe_256px_Kim_Drop_L2_150E.h5')
# model.load_weights('./tmp/vgg_b_fer2013/vgg_fer_r3/Best-weights-my_model-013-0.6227.hdf5')

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
        epochs=128,
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
filename = 'model_train_vgg_fer.csv'
csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

# Export best weights
fp = "./tmp/vgg_b_fer2013/Best-weights-my_model-{epoch:03d}-{val_acc:.4f}.hdf5"
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

H = model.fit(
    X_train, Y_train,
    batch_size=config.BATCH_SIZE,
    validation_data=(X_val, Y_val),
    epochs=config.NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=config.BATCH_SIZE)
print(classification_report(Y_test.argmax(axis=1),
      predictions.argmax(axis=1), target_names=config.CLASSES))
print("-- %s seconds --" % (time.time() - start_time))

# H = model.fit_generator(
#     aug.flow(X_train, Y_train, batch_size=config.BATCH_SIZE),
#     validation_data=(X_test, Y_test),
#     steps_per_epoch=X_train.shape[0] // config.BATCH_SIZE,
#     epochs=config.NUM_EPOCHS,
#     callbacks=callbacks_list,
#     verbose=1)

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
plt.savefig(os.path.sep.join(["output", "conf_mtx_vgg_fer.png"]))

# Save normalized confusion matrix
plot_confusion_matrix(y_trueskl, y_predskl, normalize=True)
plt.savefig(os.path.sep.join(["output", "conf_mtx_vgg_fer_norm.png"]))


