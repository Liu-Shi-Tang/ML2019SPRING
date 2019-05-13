from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras import backend as K
from PIL import Image
import os
import sys
import numpy as np


def getModel (shape_in=(32,32,3)) :
  input_img = Input(shape=shape_in)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)                 # width,height /= 2
  x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)               # width,height /= 2

  # shape of encode is (32/4,32/4,3)

  x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded) 
  x = UpSampling2D((2, 2))(x)                         # width,height *= 2
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)                         # width,height *= 2
  decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

  # shape of decode is (32,32,3)

  autoencoder = Model(input_img,decoded)
  autoencoder.compile(
      optimizer='adadelta', 
      loss='mean_squared_error',
      metrics=['accuracy'])
  
  return encoded,decoded,autoencoder


inputDir = sys.argv[1] 

# extract file name
pngList = [ file for file in os.listdir(inputDir) if file.endswith('.jpg')]


pictures = [ np.asarray(Image.open(inputDir + pic)) for pic in pngList ]
pictures = np.array(pictures)


pictures.astype('float32') / 255


encoder, decoder, autoencoder = getModel((32,32,3))

autoencoder.summary()


csv_logger = CSVLogger('log.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, min_delta=1e-4, min_lr=1e-6)
checkpoint = ModelCheckpoint(filepath='best.h5', monitor='val_acc', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

history = autoencoder.fit(x=pictures,y=pictures,batch_size=256,epochs=50,validation_split=0.1,shuffle=True,callbacks=[learning_rate, checkpoint, early_stop, csv_logger])


