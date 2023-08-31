import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import GlobalAveragePooling2D
#from tensorflow.keras.callbacks import EarlyStopping

cwd = os.getcwd()
train_target_path = os.path.join(cwd,'YOLO/train/reshaped_train' )
test_target_path = os.path.join(cwd,'YOLO/test/reshaped_test' )
# Load data
image_folder_train = train_target_path
all_image_files_train = sorted([os.path.join(image_folder_train ,f) for f in os.listdir(image_folder_train)])

image_folder_test = test_target_path
all_image_files_test = sorted([os.path.join(image_folder_test, f) for f in os.listdir(image_folder_test)])

# Select the top 100 files for training and the next 50 for testing
train_image_files = all_image_files_train
test_image_files = all_image_files_test

train = pd.read_csv('archive/sign_mnist_train.csv')
train_labels_list = train['label'].values

test = pd.read_csv('archive/sign_mnist_test.csv')
test_labels_list = test['label'].values

# # Shuffle the indices
# indices = np.arange(len(all_image_files))
# np.random.shuffle(indices)
#
# # Select shuffled files for training and testing
# train_indices = indices[:100]
# test_indices = indices[100:150]
#
# train_image_files = [all_image_files[i] for i in train_indices]
# test_image_files = [all_image_files[i] for i in test_indices]
# train_labels = labels_list[train_indices]
# test_labels = labels_list[test_indices]

# Read and preprocess images
train_images = [cv2.imread(f) for f in train_image_files]
train_images = [cv2.resize(img, (224, 224)) for img in train_images]

test_images = [cv2.imread(f) for f in test_image_files]
test_images = [cv2.resize(img, (224, 224)) for img in test_images]

# Convert to NumPy arrays
X_train = np.array(train_images)
y_train = np.array(train_labels_list)

X_test = np.array(test_images)
y_test = np.array(test_labels_list)



# Number of classes
num_classes = 25


# Model definition
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define early stopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compile the model
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
#, callbacks=[early_stopping])

# Save the model
model.save('Model_save_efficientb0/')

