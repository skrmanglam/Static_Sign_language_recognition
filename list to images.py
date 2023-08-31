import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load your dataset into a DataFrame
train = pd.read_csv('archive/sign_mnist_test.csv')

train = train.drop(train.columns[0], axis = 1)

output_dir = 'YOLO/test/reshaped_test'
os.makedirs(output_dir, exist_ok=True)

for index, row in train.iterrows():
    img_array = row.values.reshape(28, 28)
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(os.path.join(output_dir, f'image_{index}.png'))


