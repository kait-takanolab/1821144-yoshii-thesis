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
from keras.models import load_model
model = load_model('my_model.h5')
print(model.summary())
plot_model(model, to_file='model.png')



