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
from sklearn.decomposition import PCA

inputDir = sys.argv[1] 

def getModel (shape_in=(32,32,3)) :
  input_img = Input(shape=shape_in)
  x = Conv2D(32, (3, 3) , strides = (1,1), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), strides = (2,2) , padding='same')(x)                           # width,height /= 2
  x = Conv2D(32, (3, 3) , strides = (1,1), activation='relu', padding='same')(x)
  # x = MaxPooling2D((2, 2), strides = (2,2) , padding='same')(x)                           # width,height /= 2
  # shape is (32/4,32/4,3)
  x = Flatten()(x)
  # x = Dense(1024 , activation='relu')(x)
  encoded = Dense(1024 , activation='relu')(x)
  # shape of encoded is 512


  x = Dense(4096 , activation='relu')(encoded)
  # x = Dense(4096, activation='relu')(x)
  x = Reshape((8,8,64))(x)
  x = UpSampling2D((2, 2))(x)                                                             # width,height *= 2
  x = Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')(x) 
  x = UpSampling2D((2, 2))(x)                                                             # width,height *= 2
  x = Conv2D(16, (3, 3), strides=(1,1), padding='same', activation='relu')(x)
  decoded = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

  # shape of decode is (32,32,3)

  autoencoder = Model(input_img,decoded)
  autoencoder.compile(
      optimizer='rmsprop', 
      loss='mae',
      metrics=['accuracy'])
  
  return input_img,encoded,decoded,autoencoder


pngList = []
for i in range(1,40001,1) :
  pngList.append("%06d" %i)

pictures = [ np.asarray(Image.open( os.path.join(inputDir,(pic + '.jpg'))  )) for pic in pngList ]
pictures = np.array(pictures)


pictures = pictures.astype('float32') / 255


inputLayer, encoder, decoder, autoencoder = getModel((32,32,3))

autoencoder.summary()


csv_logger = CSVLogger('log.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='loss',factor = 0.2, patience=5, verbose=1, mode='auto', min_delta=1e-4,cooldown=0, min_lr=1e-8)
checkpoint = ModelCheckpoint(filepath='best.h5', monitor='loss', verbose=1, save_best_only=True,save_weights_only=False,mode='auto',period=1)
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto',min_delta=0.0002 )

history = autoencoder.fit(x=pictures,y=pictures,batch_size=256,epochs=5000,shuffle=True,callbacks=[learning_rate, checkpoint, early_stop, csv_logger])



myencoder = Model(inputLayer, encoder)
myencoder.save("encoder.h5")


processInputImgs = myencoder.predict(pictures)

processInputImgs = processInputImgs.reshape((len(pictures),-1))

seed = 40666888

pca = PCA(n_components=96, whiten=True, random_state=seed)
pca.fit(processInputImgs)
processInputImgs = pca.transform(processInputImgs)

result = KMeans(n_clusters = 2, max_iter=5000, n_init=50 ,verbose = 0 , n_jobs=-1 , random_state=seed).fit(processInputImgs)


