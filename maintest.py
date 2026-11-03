import cv2
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np

model = load_model('Braintumor10epochs.h5')
image = cv2.imread('D:\\archive\\pred\\pred10.jpg')

img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)

input_img = np.expand_dims(img,axis=0)

result = model.predict(input_img)
print(result)