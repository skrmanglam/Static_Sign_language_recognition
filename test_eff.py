import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
import timeit
from sklearn import metrics
import tensorflow as tf
#from sklearn.preprocessing import LabelBinarizer
import os

# # reading in the test data
# test = pd.read_csv('archive/sign_mnist_test.csv')



# # Creating separate labels and testing data.
# y_test = test.iloc[:,0]
# x_test = test.drop(['label'], axis = 1)

#print(os.getcwd())

# Load data
image_folder = "YOLO/test/reshaped_test"
all_image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
test_image_files = all_image_files

test = pd.read_csv('archive/sign_mnist_test.csv')
labels_list = test['label'].values

# # Checking number of unique values thus number of classes.
#
# label_binarizer = LabelBinarizer()
# test_labels = label_binarizer.fit_transform(y_test)

# #Grayscale-Normalizing the data or faster convergence
# x_test = x_test / 255

#Reshaping to create 2828 image from 784 pixel values
#
# x_test = x_test.values.reshape(-1,28,28,1)

test_images = [cv2.imread(f) for f in all_image_files]
test_images = [cv2.resize(img, (224, 224)) for img in test_images]

X_test = np.array(test_images)
y_test = np.array(labels_list)

# Number of classes
#num_classes = 25


model = tf.keras.models.load_model('Model_save_efficientb0/')



def inference():
    y_pred = model.predict(X_test)
    return y_pred

# Number of loops
n_loops = 5
# Timer setup
t = timeit.Timer("inference()", globals=globals())
# Time it
elapsed_time = t.timeit(n_loops) / n_loops
print(f"Average inference time: {elapsed_time:.4f} seconds")


y_pred_prob = inference()
# Convert softmax output to predicted labels
y_pred_labels = np.argmax(y_pred_prob, axis=1)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"Accuracy: {accuracy}")
#y_pred = np.array(y_pred)

#y_pred = np.where(y_pred>0.5,1,0)
#print("y_test shape:", y_test.shape)
#print("y_pred shape:", y_pred.shape)

f1 = metrics.f1_score(y_test, y_pred_labels, average='weighted')
print(f"F1 Score: {f1}")




#y_pred = np.argmax(y_pred, axis=1)
print('Actual Label', y_test[0])
print('Predicted Label', y_pred_labels[0])

#Selecting a random sample from the test set
# plt.title("Predicted Class {}\nActual Class {}".format(y_pred[0], y_test[0]), fontsize=10)
# plt.imshow(x_test[0].reshape(28,28), cmap = 'gray')
# plt.show()