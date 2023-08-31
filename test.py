import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import timeit
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# reading in the test data
test = pd.read_csv('archive/sign_mnist_test.csv')

# Creating separate labels and testing data.
y_test = test.iloc[:,0]
x_test = test.drop(['label'], axis = 1)

# Checking number of unique values thus number of classes.

label_binarizer = LabelBinarizer()
test_labels = label_binarizer.fit_transform(y_test)

#Grayscale-Normalizing the data or faster convergence
x_test = x_test / 255

#Reshaping to create 2828 image from 784 pixel values

x_test = x_test.values.reshape(-1,28,28,1)

model = tf.keras.models.load_model('Model_save_vr1/')

# Number of loops
n_loops = 5

# Timer setup
t = timeit.Timer("inference()", globals=globals())

def inference():
    y_pred = model.predict(x_test)
    return y_pred
y_pred = inference()
accuracy_score(test_labels, y_pred.round())
y_pred = np.array(y_pred)

y_pred = np.where(y_pred>0.5,1,0)
f1s = metrics.f1_score(test_labels, y_pred, average = 'weighted')
print('F1 score is = ', f1s)

# Time it
elapsed_time = t.timeit(n_loops) / n_loops
print(f"Average inference time: {elapsed_time:.4f} seconds")


y_pred = np.argmax(y_pred, axis=1)
print(len(y_pred))
print('Actual Label',y_test[0])
print('Predicted Label', y_pred[0])

#Selecting a random sample from the test set
# plt.title("Predicted Class {}\nActual Class {}".format(y_pred[0], y_test[0]), fontsize=10)
# plt.imshow(x_test[0].reshape(28,28), cmap = 'gray')
# plt.show()