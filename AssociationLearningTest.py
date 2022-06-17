
# use smaller dataset jaffe plus ck
# check the code with prints

# Packages import
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils

from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import SGD,RMSprop,adam
from keras.preprocessing.image import ImageDataGenerator

# Read images

# Define data path
data_path = './mergedDataset/'
data_dir_list = os.listdir(data_path)

img_rows = 256
img_cols = 256
num_channel = 1

num_epoch = 10

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # input_img_resize = cv2.resize(input_img, (48, 48))
        # img_data_list.append(input_img_resize)
        img_data_list.append(input_img)

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

labels[0:74] = 0  # 75
labels[74:162] = 1  # 87
labels[162:220] = 2  # 57
labels[220:321] = 3  # 100
labels[321:470] = 4  # 148
labels[470:530] = 5  # 59
labels[530:] = 6  # 113

print(len(labels))


names = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']


def getLabel(id):
    return ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE'][id]

# Class label to one-hot encoding


Y = np_utils.to_categorical(labels, num_classes)
# print(Y)
# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# print(x, "\n", y)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

# print(len(X_train), " and\n", len(X_test))
# print(len(y_train), " and\n", len(y_test))
# CNN layer set
# Defining the model

input_shape = img_data[0].shape
print(input_shape)
# print(img_data)

model = Sequential()

# Feature Extraction
model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(120, (5, 5)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Classification
model.add(Flatten())
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


# Model Save
# model.save_weights('model_weights_run_two_ckjaffe_256px_2_12.h5')
# model.save('model_keras_run_two_ckjaffe_256px_2_12.h5')

# Model Load
# from keras.models import load_model
# model = load_model('model_keras.h5')
# #model.load_weights('model_weights.h5')

# View model configuration
# model.get_weights()
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

from keras import callbacks
filename = 'model_train_new.csv'
filepath = "Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

csv_log = callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log, checkpoint]
callbacks_list = [csv_log]

# Model training
hist = model.fit(X_train, y_train, batch_size=7, epochs=30, verbose=1, validation_data=(X_test, y_test), callbacks=callbacks_list)

# visualizing losses and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']

epochs = range(len(train_acc))

plt.plot(epochs, train_loss, 'r', label='train_loss')
plt.plot(epochs, val_loss, 'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, 'r', label='train_acc')
plt.plot(epochs, val_acc, 'b', label='val_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()

# Evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

res = model.predict_classes(X_test[:9])
plt.figure(figsize=(10, 10))

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_test[i],cmap=plt.get_cmap('gray'))
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=14)
# show the plot
plt.show()

from sklearn.metrics import confusion_matrix
results = model.predict_classes(X_test)
cm = confusion_matrix(np.where(y_test == 1)[1], results)
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Test with a new image
testimg_data_list = []
test_img = cv2.imread('TestImages/foto_vasilkova_profi_orez.jpg', True)
# test_img = cv2.imread('mergedDataset/HAPPY/happy20.jpg', True)
test_img_resize = cv2.resize(test_img, (256, 256))
testimg_data_list.append(test_img_resize)
testimg_data = np.array(testimg_data_list)
testimg_data = testimg_data.astype('float32')
testimg_data = testimg_data / 255
testimg_data.shape

print("test image original shape", testimg_data[0].shape)
print("image original shape", img_data[0].shape)

results = model.predict_classes(testimg_data)
plt.imshow(test_img, cmap=plt.get_cmap('Set2'))
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.xlabel('prediction = %s' % getLabel(results[0]), fontsize=25)

