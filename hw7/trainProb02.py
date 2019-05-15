from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras import backend as K
from PIL import Image
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import mean_absolute_error

def getModel (shape_in=(32,32,3)) :
  input_img = Input(shape=shape_in)
  x = Conv2D(64, (3, 3) , strides = (1,1), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), strides = (2,2) , padding='same')(x)                           # width,height /= 2
  x = Conv2D(64, (3, 3) , strides = (1,1), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides = (2,2) , padding='same')(x)                           # width,height /= 2
  # shape is (32/4,32/4,3)
  x = Flatten()(x)
  x = Dense(512 , activation='relu')(x)
  encoded = Dense(256 , activation='relu')(x)
  # shape of encoded is 256


  x = Dense(512 , activation='relu')(encoded)
  x = Dense(3136, activation='relu')(x)
  x = Reshape((7,7,64))(x)
  x = UpSampling2D((2, 2))(x)                                                             # width,height *= 2
  x = Conv2DTranspose(64, (3, 3), strides=(1,1), activation='relu', padding='same')(x) 
  x = UpSampling2D((2, 2))(x)                                                             # width,height *= 2
  x = Conv2DTranspose(64, (3, 3), strides=(1,1), activation='relu', padding='same')(x)
  decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

  # shape of decode is (32,32,3)

  autoencoder = Model(input_img,decoded)
  autoencoder.compile(
      optimizer='rmsprop', 
      loss='mae',
      metrics=['accuracy'])
  
  return input_img,encoded,decoded,autoencoder


