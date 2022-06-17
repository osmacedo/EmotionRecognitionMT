
# Packages import
import os
import cv2
import time
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.utils import np_utils
from keras.models import load_model

# Read images

# Define data path for testing
data_path_test = './ferBigTestDataset/'
data_dir_list_test = os.listdir(data_path_test)

img_data_list_test = []

for dataset in data_dir_list_test:
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


# Model Load

model = load_model('./load_models_weights/fer_2013_kim_r3/model_keras_fer_48px_Kim_v1.h5')
model.load_weights('./load_models_weights/fer_2013_kim_r3/Best-weights-my_model-048-0.6018.hdf5')

# View model configuration
# model.get_weights()
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])

# Evaluating the model
start_time = time.time()
print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=256)
print(classification_report(Y_test.argmax(axis=1),
      predictions.argmax(axis=1), target_names=names))
print("-- %s seconds --" % (time.time() - start_time))


score = model.evaluate(X_test, Y_test, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
test_all_imgs = X_test[0:]
print("Shape of test image: ", test_image.shape)
print("Shape all images:", test_all_imgs.shape)

print("Prediction of first images: ", model.predict(test_image))
print("Ground truth In one-hot: ", Y_test[0:1])
print("Predicted class: ", model.predict_classes(test_image))

print("Prediction all: ", model.predict(test_all_imgs))
print("Predicted class all: ", model.predict_classes(test_all_imgs))
cnn_test_images_fer = model.predict(test_all_imgs)

