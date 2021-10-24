#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pydub import AudioSegment
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


name ="ウルトラ大掃除.mp3"
path = os.getcwd()


# In[3]:


sound = AudioSegment.from_file(name)


# In[4]:


samples = np.array(sound.get_array_of_samples())
sample = samples[::sound.channels]


# In[5]:


#フーリエ変換
spec = np.fft.fft(sample)


# In[6]:


freq = np.fft.fftfreq(sample.shape[0],1.0/sound.frame_rate)


# In[7]:


#窓幅
w = 2000
#刻み
s = 500


# In[8]:


#スペクトル格納用
ampList = []

#刻みずつずらしながら窓幅分のデータをフーリエ変換する
for i in range(int((sample.shape[0]- w) / s)):
    data = sample[i*s:i*s+w]
    spec = np.fft.fft(data)
    spec = spec[:int(spec.shape[0]/2)]
    spec[0] = spec[0] / 2
    ampList.append(np.abs(spec))


#周波数は共通なので１回だけ計算（縦軸表示に使う）  
freq = np.fft.fftfreq(data.shape[0], 1.0/sound.frame_rate)
freq = freq[:int(freq.shape[0]/2)]

#時間も共通なので１回だけ計算（横軸表示に使う）
time = np.arange(0, i+1, 1) * s / sound.frame_rate

#numpyの配列にしておく
ampList = np.array(ampList)


# In[9]:


df_amp = pd.DataFrame(data=ampList, index=time, columns=freq)


# In[10]:


#seabornのheatmapを使う
a = plt.figure(figsize=(20, 6))
l = sns.heatmap(data=np.log(df_amp.iloc[:, :100].T), 
            xticklabels=1000, 
            yticklabels=10, 
            cmap=plt.cm.gist_rainbow_r,
           )
l.set_ylabel("freq[Hz]",fontsize = 20)
l.set_xlabel("time[s]",fontsize = 20)


# In[14]:


a.savefig("%s\/%s.png"%(path, name),bbox_inches="tight")


# In[ ]:




