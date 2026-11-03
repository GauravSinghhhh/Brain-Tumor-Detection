import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras .layers import Conv2D , MaxPooling2D , Activation , Dropout , Flatten , Dense

image_directory ='datasets/'
dataset = []
label = []

INPUT_SIZE = 64

no_brain_tumor = os.listdir(image_directory+'no/')
yes_brain_tumor = os.listdir(image_directory+'yes/')
#print(len(no_brain_tumor))

#Pre-Processing

for i, image_name in enumerate(no_brain_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image , 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_brain_tumor):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image , 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

#print(dataset)
#print(len(label))

dataset = np.array(dataset)
label = np.array(label)

#print(dataset)

x_train , x_test , y_train , y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

x_train = normalize(x_train , axis=1)
x_test = normalize(x_test , axis=1)

#ModelBuilding

model = Sequential()
model.add(Conv2D(32 , (3,3) , input_shape = (INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32 , (3,3) , kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32 , (3,3) , kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size=16, epochs=15, validation_data=(x_test, y_test), shuffle=False)

model.save('Braintumor10epochs.h5')


# Plotting the accuracy and loss curves
'''import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

# Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()
'''