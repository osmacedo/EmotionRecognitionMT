
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
    print('Loading images from dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # input_img_resize = cv2.resize(input_img, (48, 48))
        # img_data_list.append(input_img_resize)
        img_data_list.append(input_img)

    print('Loaded images from dataset-' + '{}\n'.format(dataset))

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255
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
# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

# Model Load

model = load_model('./load_models_weights/merged_kim_512_r5/model_keras_ckjaffe_256px_Kim_CLR_256E_R1.h5')
model.load_weights('./load_models_weights/merged_kim_512_r5/Best-weights-my_model-173-0.8049.hdf5')

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
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),
      predictions.argmax(axis=1), target_names=names))
print("-- %s seconds --" % (time.time() - start_time))


score = model.evaluate(X_test, y_test, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
test_all_imgs = X_test[0:]
print("Shape of test image: ", test_image.shape)
print("Shape all images:", test_all_imgs.shape)

print("Prediction of first images: ", model.predict(test_image))
print("Ground truth In one-hot: ", y_test[0:1])
print("Predicted class: ", model.predict_classes(test_image))

print("Prediction all: ", model.predict(test_all_imgs))
print("Predicted class all: ", model.predict_classes(test_all_imgs))
cnn_test_images = model.predict(test_all_imgs)

