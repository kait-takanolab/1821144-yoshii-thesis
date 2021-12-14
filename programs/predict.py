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
import time

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras
from keras.models import Sequential,load_model

import warnings
warnings.filterwarnings('ignore')


# In[5]:
t1 = time.time() 

cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'Electric Folk HipHop International Latin Metal Noise Pop Rock Punk'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'{g}'):
        songname = f'{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()


# In[8]:


header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


# In[9]:


file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'Electric Folk HipHop International Latin Metal Noise Pop Rock Punk'.split()
for g in genres:
    for filename in os.listdir(f'{g}'):
        songname = f'{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            


# In[10]:


data = pd.read_csv('data.csv')
data.head()


# In[11]:


data.shape


# In[12]:


data = data.drop(['filename'],axis=1)


# In[13]:


genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)


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
model.save('./my_model.h5')
model = load_model('./my_model.h5')

predictions = model.predict(X_test)


# In[25]:


predictions[0].shape


# In[26]:


np.sum(predictions[0])


# In[27]:

prediction = np.argmax(predictions[0])
print(genres[prediction])
t2 = time.time()
elapsed_time = t2-t1
print(f"経過時間：{elapsed_time}")

# In[ ]:




