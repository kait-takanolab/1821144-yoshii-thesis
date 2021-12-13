#!/usr/bin/env python
# coding: utf-8

# In[4]:


# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')


# In[5]:


cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
        songname = f'C:\Users\1821144\Documents\GitHub\1821144-yoshii-thesis\programs\Electric\001929.mp3'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        pic = plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()


# In[8]:



# In[9]:





# In[14]:


scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[16]:


len(y_train)


# In[17]:


len(y_test)


# In[18]:


X_train[10]


# In[19]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


# In[20]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[21]:


history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)


# In[22]:


test_loss, test_acc = model.evaluate(X_test,y_test)


# In[23]:


print('test_acc: ',test_acc)


# In[24]:


predictions = model.predict(X_test)


# In[25]:


predictions[0].shape


# In[26]:

a
np.sum(predictions[0])


# In[27]:

model.save("my_model")
prediction = np.argmax(predictions[0])
print(genres[prediction])

# In[ ]:




