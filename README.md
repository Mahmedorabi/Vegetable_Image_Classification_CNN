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
![data visual vege](https://github.com/Mahmedorabi/Vegetable_Image_Classification_CNN/assets/105740465/30458995-10fe-4342-b863-0b5edb00cd81)


## Data Preprocessing 
```python
data_generator=ImageDataGenerator(rescale=1/255)
train_data=data_generator.flow_from_directory(train_path
                                              ,target_size=(224,224),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)

test_data=data_generator.flow_from_directory(test_path,
                                             batch_size=32,
                                             target_size=(224,224),
                                             class_mode='categorical',
                                             shuffle=True)

validation_data=data_generator.flow_from_directory(validation_path,
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle=True,
                                                   target_size=(224,224))
```
## Build Model
**1.Add Layers**
```python
# Add Layers into model

# Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[224,224,3]))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten Layer 
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))

# Output Layer
model.add(Dense(15,activation='softmax'))
```
**2. Compile model**
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
**3. Fit model**
```python
model_hist=model.fit(train_data,
                     validation_data=validation_data,epochs=5)
```
## Model Visualization
````python
fig,ax=plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Model Visualization",fontsize=20)
ax[0].plot(model_hist.history['loss'],label='Training Loss')
ax[0].plot(model_hist.history['val_loss'],label='Testing Loss')
ax[0].set_title("Training Loss VS. Testing Loss")

ax[1].plot(model_hist.history['accuracy'],label='Training Accuracy')
ax[1].plot(model_hist.history['val_accuracy'],label='Testing Accuracy')
ax[1].set_title("Training Accuracy VS. Testing Accuracy")
plt.show()
````




![model visual Vege](https://github.com/Mahmedorabi/Vegetable_Image_Classification_CNN/assets/105740465/3c7a4a22-68e5-4381-95b7-eb0558314866)

## Usage
### To use this notebook:

**1. Ensure your dataset is correctly placed in the specified directories.**<br>
**2. Run each cell in the notebook sequentially to build and train the CNN model.**<br>
**3. Use the plot_images function to visualize the images and their categories.**

## Results
The notebook will output the accuracy and loss of the model during training and validation. It will also visualize some sample images from the dataset along with their predicted and actual labels.












