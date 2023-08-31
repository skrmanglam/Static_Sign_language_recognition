"""
Python script to perform following actions in sequence:
Load Data
Preprocess Data
Train
Save the model
Convert the model to tf-lite
"""

#imports
import numpy as np #linear algebra
import os #accessing directory structure
#import matplotlib.pyplot as plt # plotting
import pandas as pd #data loading/pre-processing
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import classification_report,confusion_matrix

"""
The block of code below helps you walk through the directory
Uncomment to view directory structure and find train files
"""
# for dirname, _, filenames in os.walk(os.getcwd()):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


"""
Load the train data
Considering it is in same directory inside archive folder
Also it expects the data to be in '.csv' format
"""
train = pd.read_csv('archive/sign_mnist_train.csv')


"""
split the dataset
"""
Y_train = train['label'].values
x_train = train.drop(['label'], axis = 1)

# Checking number of unique values thus number of classes.
unique_value = np.array(Y_train)
np.unique(unique_value)

# One-hot encoding
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(Y_train)

# Grayscale-Normalizing the data or faster convergence
x_train = x_train / 255

# Reshaping to create 28*28 image from 784 pixel values
x_train = x_train.values.reshape(-1,28,28,1)

# Creating test-train split in the test data for validation
X_train, X_val, y_train, y_val = train_test_split(x_train, labels, test_size = 0.3, random_state = 42)

# Defining tunable params
batch_size = 128
num_classes = 24
epochs = 50

"""
Model Architecture
"""
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

# compile
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# training
history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=epochs, batch_size=batch_size)

# Save the model
model.save('Model_save_vr1/')


