from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras import backend as K
from PIL import Image
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from keras.layers.normalization import BatchNormalization


def getModel (shape_in=(32,32,3)) :
  input_img = Input(shape=shape_in)
  x = Conv2D(3, (3, 3), activation='relu', padding='same')(input_img)
  x = BatchNormalization()(x)
  x = MaxPooling2D((2, 2), padding='same')(x)                           # width,height /= 2
  x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  # x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), padding='same')(x)                     # width,height /= 2
  x = Flatten()(x)
  x = Dense(32)(x)
  x = Dense(16)(x)
  encoded = Dense(8)(x)
  # shape of encode is (32/4,32/4,3)

  x = Dense(8)(encoded)
  x = Dense(64)(x)
  x = Dense(192)(x)
  x = Reshape((8,8,3))(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x) 
  # x = BatchNormalization()(x)
  x = UpSampling2D((2, 2))(x)                                           # width,height *= 2
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  # x = BatchNormalization()(x)
  x = UpSampling2D((2, 2))(x)                                           # width,height *= 2
  decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

  # shape of decode is (32,32,3)

  autoencoder = Model(input_img,decoded)
  autoencoder.compile(
      optimizer='adadelta', 
      loss='mean_squared_error',
      metrics=['accuracy'])
  
  return input_img,encoded,decoded,autoencoder


inputDir = sys.argv[1] 

# extract file name
# pngList = [ file for file in os.listdir(inputDir) if file.endswith('.jpg')]

pngList = []
for i in range(1,40001,1) :
  pngList.append("%06d" %i)

pictures = [ np.asarray(Image.open(inputDir + pic + '.jpg')) for pic in pngList ]
pictures = np.array(pictures)


pictures = pictures.astype('float32') / 255


inputLayer, encoder, decoder, autoencoder = getModel((32,32,3))

autoencoder.summary()


csv_logger = CSVLogger('log.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, min_delta=1e-4, min_lr=1e-6)
checkpoint = ModelCheckpoint(filepath='best.h5', monitor='acc', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='acc', patience=5, verbose=1)

history = autoencoder.fit(x=pictures,y=pictures,batch_size=256,epochs=550,shuffle=True,callbacks=[learning_rate, checkpoint, early_stop, csv_logger])


myencoder = Model(inputLayer, encoder)

processInputImgs = myencoder.predict(pictures)

processInputImgs = processInputImgs.reshape((len(pictures),-1))

result = KMeans(n_clusters = 2).fit(processInputImgs)

print(type(result.labels_))
print(result.labels_)


testFile = sys.argv[2]

test = np.genfromtxt( testFile , delimiter= ',' , dtype=int , skip_header=1 )
i1 , i2 = test[:,1] , test[:,2]
print(len(result.labels_))

n_z = 0
n_o = 0

with open('result.csv','w') as f :
  f.write('id,label\n')
  for i in range(len(test)) :
    if result.labels_[i1[i]-1] == result.labels_[i2[i]-1] :
      f.write(str(i)+ ',' + str(1) + '\n')
      n_o += 1
    else :
      f.write(str(i)+ ',' + str(0) + '\n')
      n_z += 1

print(str(n_z) + ',' + str(n_o) )

for idx in range(50) :
  p = autoencoder.predict(np.expand_dims(pictures[idx],axis=0))
  p = np.clip(p,0,1)
  p *= 255
  p = p.astype('uint8')
  p = Image.fromarray(p[0])
  p.save('out{}.jpg'.format(idx))
  
  x = pictures[idx]
  x *= 255
  x = np.clip(x,0,255)
  x = x.astype('uint8')
  x = Image.fromarray(x)
  x.save('ori{}.jpg'.format(idx))

