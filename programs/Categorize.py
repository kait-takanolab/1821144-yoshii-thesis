#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#モデルの作成プログラム

import os 
import os.path
import librosa
import numpy as np

path = os.getcwd()
os.chdir('%s\\music'%path)
def load(dir_path, label):
    n_mfcc = 20
    genre_x = np.zeros((0,n_mfcc))
    genre_y = np.zeros((0,1), dtype= 'int')
    
    files = os.listdir(dir_path)
    for i,file in enumerate(files):
        file_path = dir_path + file
        y, sr =librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mean = np.mean(mfcc, axis = 1)
        genre_x = np.vstack((genre_x,mean))
        genre_y = np.vstack((genre_y, label))
        
        print(f'{i+1}/{len(files)} loaded: {file_path}')
    
    return genre_x, genre_y

if __name__ == '__main__':
    folk_x, folk_y = load('Folk\\', 0)
    electronic_x, electronic_y = load('Electric\\', 1)
    
    X = np.r_[folk_x, electronic_x]
    Y = np.r_[folk_y, electronic_y]
    
    np.save('x.npy', X)
    np.save('y.npy', Y)


# In[16]:


#学習・評価プログラム

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

x = np.load('x.npy')
y = np.load('y.npy')


x_train, x_test, y_train,y_test = train_test_split(x, y, train_size=0.8)

model = Sequential()
model.add(Dense(256, activation='relu',input_shape=(20,)))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=128)


score = model.evaluate(x_test, y_test, batch_size=128)
print(f'loss* {score[0]}, accuracy: {score[1]}')


# In[ ]:




