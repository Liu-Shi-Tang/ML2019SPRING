from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Model, load_model
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

pngList = []
for i in range(1,40001,1) :
  pngList.append("%06d" %i)

pictures = [ np.asarray(Image.open(inputDir + pic + '.jpg')) for pic in pngList ]
pictures = np.array(pictures)


pictures = pictures.astype('float32') / 255

myencoder = load_model("encoder.h5")


processInputImgs = myencoder.predict(pictures)

processInputImgs = processInputImgs.reshape((len(pictures),-1))

seed = 43095

pca = PCA(n_components=96, whiten=True, random_state=seed)
pca.fit(processInputImgs)
processInputImgs = pca.transform(processInputImgs)

result = KMeans(n_clusters = 2, max_iter=5000, n_init=500 ,verbose = 0 , n_jobs=-1 , random_state=seed).fit(processInputImgs)


testFile = sys.argv[2]

test = np.genfromtxt( testFile , delimiter= ',' , dtype=int , skip_header=1 )
i1 , i2 = test[:,1] , test[:,2]


ans_label = np.load('mylabel.npy')
count = 0 
for i in range(len(ans_label)) :
  if ans_label[i] == result.labels_[i] :
    count += 1
print("acc: ",count/len(ans_label))


with open('result.csv','w') as f :
  f.write('id,label\n')
  for i in range(len(test)) :
    if result.labels_[i1[i]-1] == result.labels_[i2[i]-1] :
      f.write(str(i)+ ',' + str(1) + '\n')
    else :
      f.write(str(i)+ ',' + str(0) + '\n')




