# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14TBqqvHv53duTvnlU4FnRiAFsfw$"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictionsimport numpy as np
model = VGG16(weights='imagenet')
img_path = 'yoshii.jpg'
img = image.load_img(img_path,target_size=(224,224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)
print('Predicted:',decode_predictions(preds,top=3)[0])s