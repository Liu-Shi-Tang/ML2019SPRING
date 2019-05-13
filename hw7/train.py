from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras import backend as K
from PIL import Image
import os
import sys
import numpy as np
from sklearn.cluster import KMeans

def getModel (shape_in=(32,32,3)) :
  input_img = Input(shape=shape_in)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2), padding='same')(x)                           # width,height /= 2
  x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2), padding='same')(x)                     # width,height /= 2

  # shape of encode is (32/4,32/4,3)

  x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded) 
  x = UpSampling2D((2, 2))(x)                                           # width,height *= 2
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
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
pngList = [ file for file in os.listdir(inputDir) if file.endswith('.jpg')]


pictures = [ np.asarray(Image.open(inputDir + pic)) for pic in pngList ]
pictures = np.array(pictures)


pictures = pictures.astype('float32') / 255


inputLayer, encoder, decoder, autoencoder = getModel((32,32,3))

autoencoder.summary()


csv_logger = CSVLogger('log.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, min_delta=1e-4, min_lr=1e-6)
checkpoint = ModelCheckpoint(filepath='best.h5', monitor='val_acc', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose=1)

history = autoencoder.fit(x=pictures,y=pictures,batch_size=256,epochs=50,validation_split=0.1,shuffle=True,callbacks=[learning_rate, checkpoint, early_stop, csv_logger])


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

with open('result.csv','w') as f :
  f.write('id,label\n')
  for i in range(len(test)) :
    if result.labels_[i1[i]-1] == result.labels_[i2[i]-1] :
      f.write(str(i)+ ',' + str(1) + '\n')
    else :
      f.write(str(i)+ ',' + str(0) + '\n')


# p = autoencoder.predict(np.expand_dims(pictures[2],axis=0))
# p = np.clip(p,0,1)
# p *= 255
# p = p.astype('uint8')
# p = Image.fromarray(p[0])
# p.save('out.jpg')

