# Vegetable_Image_Classification_CNN

 This project demonstrates a Convolutional Neural Network (CNN) based approach to classify vegetable images. The steps include data preprocessing, visualization, and model training using TensorFlow and Keras.

## Requirements

To run this project, you need the following libraries installed:

- `matplotlib`
- `numpy`
- `tensorflow`
- `os`

You can install these requirements using:

```bash
pip install matplotlib numpy tensorflow
```

The dataset consists of images of various vegetables, divided into three categories: training, testing, and validation. The dataset should be structured as follows:
```
D:/Project AI/Vegetable Images/
    ├── train/
    │   ├── Bean/
    │   ├── Bitter_Gourd/
    │   ├── ...
    ├── test/
    │   ├── Bean/
    │   ├── Bitter_Gourd/
    │   ├── ...
    ├── validation/
        ├── Bean/
        ├── Bitter_Gourd/
        ├── ...
```
## Notebook Structure
 **1. Import Libraries**
 ```python
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
```
**2.Read Folders Path**
```python
train_path = 'D:/Project AI/Vegetable Images/train'
test_path = 'D:/Project AI/Vegetable Images/test'
validation_path = 'D:/Project AI/Vegetable Images/validation'
```
**3. Return Classes Name
```python
class_name = os.listdir(train_path)
class_name
```
**4.Visualization**
```python
def plot_images(class_name):
    plt.figure(figsize=(20,20))
    for i, category in enumerate(class_name):
        image_path = train_path + '/' + category
        image_in_folder = os.listdir(image_path)
        first_image = image_in_folder[0]
        first_image_path = image_path + '/' + first_image
        img = image.load_img(first_image_path)
        img_array = image.img_to_array(img) / 255
        plt.subplot(4,4,i+1)
        plt.imshow(img_array)
        plt.title(category, fontsize=20)
        plt.axis('off')
    plt.show()
```

















