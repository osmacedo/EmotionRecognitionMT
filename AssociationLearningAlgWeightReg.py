
# plot the accuracies and loss
# create confusion matrix

# Packages import
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the number of classes
num_classes = 7

# Association algorithm
X = np.identity(num_classes)
Tmax = 1000
Tmax_sum_of_ones = 0
Tmax_vector = np.zeros(1+Tmax)  # improve: start index at 1
W_ini = np.random.randn(num_classes, num_classes)*np.sqrt(2/(num_classes+num_classes))  # He et al weights initialization
W = W_ini
delta_pos = 0.1
delta_neg = 0.1

desired_out = np.arange(0, 7)
desired_out_list = desired_out.tolist()

for t in range(Tmax):

    random_value_x = np.random.randint(num_classes)
    random_col_x = X[:, random_value_x]
    max_x_index = np.argmax(random_col_x)
    y = W.dot(random_col_x)
    k = np.argmax(y)

    if max_x_index == k:
        w = W[k][random_value_x]
        w = w + delta_pos - 0.1*w  # change in l rate
        W[k][random_value_x] = w
        Tmax_Value = 1
        Tmax_vector[t] = Tmax_Value
        Tmax_sum_of_ones += Tmax_Value

    else:
        w = W[k][random_value_x]
        w = w - delta_neg
        W[k][random_value_x] = w

    current_out = np.argmax(W, axis=0)
    current_out_list = current_out.tolist()

    if current_out_list == desired_out_list:
        break

print("W matrix: \n", W)
print("W argmax matrix: \n", np.argmax(W, axis=0))
# print("Tmax:", Tmax_vector)
print("Tmax sum: ", Tmax_sum_of_ones)
print("t:", t)

# Plot t where j = k
plt.plot(Tmax_vector)
plt.xlabel('Tmax_vector, Total count: ' + str(Tmax_sum_of_ones))
plt.show()

# Confusion matrix


