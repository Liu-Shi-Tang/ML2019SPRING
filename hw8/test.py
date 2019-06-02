import numpy as np
import sys
import keras
# For NN
from keras.models import Sequential,load_model
# For using optimizers
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
# For convolution, pooling, and connection before fully connect
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D,  BatchNormalization, LeakyReLU, DepthwiseConv2D, Flatten, AveragePooling2D, Activation
# Callbacks
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger




# read testing data
test_file = sys.argv[1] 
test_in = np.genfromtxt(fname=test_file,skip_header=1,dtype=str,delimiter=' ')
num_test = len(test_in)

for i in range(num_test) :
  test_in[i,0] = test_in[i,0].split(',')[1]

# reshaping testing data
test_in = np.reshape(test_in,(num_test,48,48,1))
test_in = test_in.astype(float)
test_in = test_in/255

# build model #####################################################################################
def getModel() :
  
  model = Sequential()
  
  model.add(Conv2D(32,(5,5),input_shape = (48,48,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2)))
  model.add(Dropout(0.02)) 
    
  model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(64,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2))) 
  model.add(Dropout(0.03)) 
  
  model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(256,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2))) 
  model.add(Dropout(0.04)) 
  
  model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='linear'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(56,(1,1), activation = 'linear', padding='same',kernel_initializer='glorot_normal'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(AveragePooling2D(pool_size=(2,2))) 
  model.add(Dropout(0.02)) 
  
  
  model.add(Flatten())
  
  model.add(Dense(20,kernel_initializer='glorot_normal'))
  # model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  
  model.add(Dense(units=7,activation='softmax'))
  myOpt = Adam(lr=0.02)
  model.compile(loss='categorical_crossentropy',optimizer=myOpt,metrics=['accuracy'])
  
  
  return model




model = getModel()
#layers = np.load('weights.npy')
#model.set_weights(layers)

model.load_weights('best.h5')
# without std
result = model.predict(test_in)



# write result
result_file = sys.argv[2]


f = open(result_file,'w')
f.write("id,label\n")
for i in range(len(result)) :
  f.write(str(i) + ',' + str(int(np.argmax(result[i]))) + '\n')

f.close()
  
print('end')


