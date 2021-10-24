#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.vgg16 import VGG16,decode_predictions,preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow
import keras


# In[2]:


filename = "dog.jpeg"
def predict(filename,size=5):
    
    img = image.load_img(filename, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    pred = model.predict(preprocess_input(x))
    results = decode_predictions(pred,top=size)[0]
    return results

model = VGG16(weights="imagenet")
results = predict(filename,10)
for result in results:
    print(result)


# In[ ]:





# In[ ]:




